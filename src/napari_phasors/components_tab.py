import contextlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import vispy.color
from matplotlib.collections import LineCollection
from matplotlib.colors import (
    LinearSegmentedColormap,
    ListedColormap,
    PowerNorm,
)
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.transforms import Affine2D
from napari.layers import Image
from napari.utils.colormaps import AVAILABLE_COLORMAPS, Colormap
from napari.utils.notifications import show_error, show_info, show_warning
from phasorpy.component import phasor_component_fit, phasor_component_fraction
from phasorpy.lifetime import (
    phasor_from_lifetime,
    phasor_semicircle_intersect,
    phasor_to_normal_lifetime,
)
from phasorpy.phasor import phasor_center
from qtpy.QtCore import QRectF, Qt
from qtpy.QtGui import QColor, QKeySequence, QPainter, QShortcut
from qtpy.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QStyle,
    QStyleOptionSlider,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from ._utils import (
    CheckableComboBox,
    HistogramWidget,
    analysis_section_stylesheet,
    make_section,
    required_component_harmonics,
    setup_primary_button,
)

if TYPE_CHECKING:
    import napari


@dataclass
class ComponentState:
    idx: int
    dot: any = None
    text: any = None
    name_edit: QLineEdit | None = None
    lifetime_edit: QLineEdit | None = None
    g_edit: QLineEdit | None = None
    s_edit: QLineEdit | None = None
    select_button: QPushButton | None = None
    number_label: QLabel | None = None
    text_offset: tuple[float, float] = (0.02, 0.02)
    label: str = "Component"
    # Layer names last used to compute the phasor center for this component.
    phasor_center_layers: list[str] = field(default_factory=list)


class ColorActionWidget(QLabel):
    def __init__(self, text, color, action, parent=None):
        super().__init__(text, parent)
        self.action = action

        # Determine CSS color representation
        css_color = "black"
        if color is not None:
            if hasattr(color, 'name'):
                css_color = color.name()
            elif hasattr(color, 'getRgb'):
                r, g, b, a = color.getRgb()
                css_color = f"rgba({r}, {g}, {b}, {a/255.0})"
            elif (
                isinstance(color, (tuple, list, np.ndarray))
                and len(color) >= 3
            ):
                if all(
                    isinstance(c, (float, np.float32, np.float64))
                    for c in color[:3]
                ) and any(c <= 1.0 for c in color[:3]):
                    r, g, b = [int(c * 255) for c in color[:3]]
                    a = float(color[3]) if len(color) > 3 else 1.0
                    css_color = f"rgba({r}, {g}, {b}, {a})"
                else:
                    r, g, b = color[:3]
                    a = float(color[3]) if len(color) > 3 else 255.0
                    css_color = f"rgba({r}, {g}, {b}, {a/255.0})"
            else:
                css_color = str(color)

        self.setStyleSheet(f"""
            QLabel {{
                color: {css_color};
                font-weight: bold;
                padding: 6px 20px;
                background-color: transparent;
                font-size: 11px;
            }}
            QLabel:hover {{
                background-color: #3b82f6;
                color: white;
            }}
        """)
        self.setMouseTracking(True)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.action.trigger()
            parent = self.parent()
            while parent:
                if isinstance(parent, QMenu):
                    parent.close()
                    break
                parent = parent.parent()


class CenterFillSlider(QSlider):
    """Horizontal slider whose colored fill originates from the center (0).

    A standard :class:`QSlider` fills its groove from the minimum edge to the
    handle, which is misleading for a signed value centered at zero. This
    subclass paints the fill from the zero position to the handle instead, so
    dragging left of centre reads as negative and right as positive.
    """

    fill_color = QColor("#3b82f6")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Neutralise the default directional groove fill so only the
        # centre-anchored fill drawn below is visible.
        self.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 4px;
                background: #c7ccd4;
                border-radius: 2px;
            }
            QSlider::sub-page:horizontal,
            QSlider::add-page:horizontal {
                background: #c7ccd4;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                width: 12px;
                margin: -5px 0;
                border-radius: 6px;
                background: #6b7280;
            }
            """)

    def paintEvent(self, event):
        super().paintEvent(event)
        span = self.maximum() - self.minimum()
        if span == 0:
            return

        opt = QStyleOptionSlider()
        self.initStyleOption(opt)
        groove = self.style().subControlRect(
            QStyle.CC_Slider, opt, QStyle.SC_SliderGroove, self
        )
        handle = self.style().subControlRect(
            QStyle.CC_Slider, opt, QStyle.SC_SliderHandle, self
        )

        # Map the zero value to a pixel the same way Qt positions the handle:
        # the handle centre only travels within the groove inset by half its
        # width. Using the naive groove-width mapping instead would drift the
        # fill origin away from the handle at value 0.
        handle_w = handle.width()
        available = groove.width() - handle_w
        zero_pos = self.style().sliderPositionFromValue(
            self.minimum(), self.maximum(), 0, available, opt.upsideDown
        )
        zero_x = groove.x() + handle_w / 2.0 + zero_pos
        handle_x = handle.center().x()

        # Overlay the fill exactly on the grey groove line. The stylesheet
        # draws the track 4px tall, centred on the groove rect; mirror that
        # here using the float centre so the blue band shares the light-grey
        # track's centre and thickness (QRect.center() floors for even
        # heights, which shifted the band up by a pixel).
        groove_thickness = 4.0
        center_y = groove.y() + groove.height() / 2.0
        left = min(zero_x, handle_x)
        right = max(zero_x, handle_x)
        rect = QRectF(
            left,
            center_y - groove_thickness / 2.0,
            max(0.0, right - left),
            groove_thickness,
        )

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(rect, self.fill_color)
        painter.end()


class PhasorCenterSelectionDialog(QDialog):
    def __init__(self, layers, parent=None, preselected=None):
        super().__init__(parent)
        self.setWindowTitle("Select Phasor Center Layers")
        self.setMinimumWidth(360)
        self.resize(360, 150)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(12)

        info_label = QLabel(
            "Select the image layer(s) to calculate the phasor center from:"
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("font-size: 11px; color: #555555;")
        layout.addWidget(info_label)

        self.layer_combo = CheckableComboBox(
            placeholder="Select layers...",
            parent=self,
            enable_primary_layer=False,
        )
        self.layer_combo.addItems(layers)

        # Pre-select only the layers that were previously used to compute the
        # phasor center for this component. By default (no prior selection)
        # nothing is checked.
        if preselected:
            checked = [name for name in preselected if name in layers]
            if checked:
                self.layer_combo.setCheckedItems(checked)

        layout.addWidget(self.layer_combo)

        # Spacer
        layout.addSpacing(4)

        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        layout.addWidget(button_box)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        self.setLayout(layout)

    def get_selected_layers(self):
        return self.layer_combo.checkedItems()


class ComponentsWidget(QWidget):
    """Widget to perform component analysis on phasor coordinates."""

    def __init__(self, viewer: "napari.viewer.Viewer", parent=None):
        super().__init__()
        self.viewer = viewer
        self.parent_widget = parent

        self.current_image_layer_name = None

        # Replace individual attributes with list
        self.components: list[ComponentState] = []

        # Keep fractions / line attributes
        self.component_line = None
        self.component_polygon = None
        self.fraction_layers = []
        self.comp1_fractions_layer = None
        self.comp2_fractions_layer = None
        self.fractions_colormap = None
        self.colormap_contrast_limits = None
        self.fractions_gamma = 1.0
        self.component_colors = [
            'magenta',
            'cyan',
            'yellow',
            'blue',
            'red',
            'green',
            'orange',
            'purple',
            'dodgerblue',
        ]
        self.component_colormap_names = [
            'magenta',
            'cyan',
            'yellow',
            'blue',
            'red',
            'green',
            'bop orange',
            'bop purple',
            'bop blue',
        ]

        # Style state
        self.label_fontsize = 10
        self.label_bold = False
        self.label_italic = False
        self.label_color = 'black'

        # Analysis type
        self.analysis_type = "Linear Projection"

        # Current harmonic
        self.current_harmonic = 1

        # Line settings
        self.show_colormap_line = True
        self.show_component_dots = True
        self.line_offset = 0.0
        self.line_width = 3.0
        self.line_alpha = 1
        self.default_component_color = 'dimgray'

        # Fraction histogram overlay settings
        self.show_fraction_histogram = False
        self.histogram_overlay_height = 0.3
        self.histogram_offset = 0.0
        self.histogram_alpha = 0.75
        self.component_histogram = None

        # Flag to prevent clearing lifetime when updating from lifetime
        self._updating_from_lifetime = False
        self._updating_settings = False  # Flag to prevent recursive updates
        self._needs_update = False  # Deferred update flag
        # Guard against recursion while mirroring slider <-> spinbox pairs.
        self._syncing_slider_spin = False
        # Guard to avoid recursion while mirroring the component selection
        # between the histogram and statistics comboboxes.
        self._syncing_component_comboboxes = False

        # Flag to track if analysis was attempted
        self._analysis_attempted = False

        # Dialog / event flags
        self.plot_dialog = None
        self.style_dialog = None
        self.drag_events_connected = False

        # Drag state
        self.dragging_component_idx = None
        self.dragging_label_idx = None

        # Select listeners state
        self._active_select_cid = None
        self._active_select_key_cid = None
        self._active_select_shortcut = None
        self._active_select_idx = None
        self._active_select_original_text = "Select"

        self.setup_ui()
        self._update_lifetime_inputs_visibility()
        self._update_analysis_options()
        self.parent_widget.harmonic_spinbox.valueChanged.connect(
            self._on_harmonic_changed
        )

    def setup_ui(self):
        """Set up the user interface for the components widget."""
        # Root layout for this widget
        root_layout = QVBoxLayout()
        self.setStyleSheet(analysis_section_stylesheet())

        # Scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        root_layout.addWidget(scroll_area)

        # Content widget inside scroll area
        content_widget = QWidget()
        scroll_area.setWidget(content_widget)

        # Previous main layout becomes content_layout
        layout = QVBoxLayout()
        content_widget.setLayout(layout)

        # Analysis type section
        analysis_box, analysis_box_layout = make_section("Analysis type")
        analysis_layout = QHBoxLayout()
        analysis_layout.addWidget(QLabel("Analysis Type:"))
        self.analysis_type_combo = QComboBox()
        self.analysis_type_combo.currentTextChanged.connect(
            self._on_analysis_type_changed
        )
        self.analysis_type_combo.setToolTip(
            "Select the type of component analysis to perform."
        )
        analysis_layout.addWidget(self.analysis_type_combo)
        analysis_layout.addStretch()
        analysis_box_layout.addLayout(analysis_layout)
        layout.addWidget(analysis_box)

        # Components section
        components_box, components_box_layout = make_section("Components")

        # Components info label
        self.components_info_label = QLabel()
        self.components_info_label.setTextFormat(Qt.RichText)
        self.components_info_label.setStyleSheet(
            "font-weight: bold; margin-top: 10px;"
        )
        self.components_info_label.setToolTip(
            "Change the harmonic spinbox number above to view components at different harmonics."
        )
        self._refresh_components_info_label()
        components_box_layout.addWidget(self.components_info_label)

        # Components inputs
        self.components_layout = QVBoxLayout()
        components_box_layout.addLayout(self.components_layout)

        # Initialize with 2 components
        for i in range(2):
            self._add_component_ui(i)

        # Move the visibility update to AFTER components are created
        self._update_lifetime_inputs_visibility()

        # Component management section
        comp_management_layout = QHBoxLayout()
        self.add_component_btn = QPushButton("Add Component")
        self.add_component_btn.clicked.connect(self._add_component)
        self.add_component_btn.setToolTip("Add a new component field.")
        comp_management_layout.addWidget(self.add_component_btn)

        self.remove_component_btn = QPushButton("Remove Component")
        self.remove_component_btn.clicked.connect(self._remove_component)
        self.remove_component_btn.setToolTip(
            "Remove the last component field."
        )
        comp_management_layout.addWidget(self.remove_component_btn)

        self.clear_components_btn = QPushButton("Clear All")
        self.clear_components_btn.clicked.connect(self._clear_components)
        self.clear_components_btn.setToolTip("Clear all component values.")
        comp_management_layout.addWidget(self.clear_components_btn)

        comp_management_layout.addStretch()
        components_box_layout.addLayout(comp_management_layout)
        layout.addWidget(components_box)

        # Calculate button (validated: greyed out until components are set)
        self.calculate_button = QPushButton("Run Component Analysis")
        self._refresh_run_button = setup_primary_button(
            self.calculate_button,
            self._components_validation,
            self._run_analysis,
            ready_tooltip="Run the selected analysis type on the defined "
            "components.",
        )
        layout.addWidget(self.calculate_button)

        # Display settings section
        display_box, display_box_layout = make_section("Display settings")

        buttons_row = QHBoxLayout()
        self.plot_settings_btn = QPushButton("Edit Line Layout...")
        self.plot_settings_btn.clicked.connect(self._open_plot_settings_dialog)
        self.plot_settings_btn.setToolTip(
            "Edit the layout of the line(s) between components."
        )
        buttons_row.addWidget(self.plot_settings_btn)

        # Label style button
        self.label_style_btn = QPushButton("Edit Component Name Layout...")
        self.label_style_btn.clicked.connect(self._open_label_style_dialog)
        self.label_style_btn.setToolTip(
            "Edit the layout of the component name labels in the plot."
        )
        buttons_row.addWidget(self.label_style_btn)

        buttons_row.addStretch()
        display_box_layout.addLayout(buttons_row)
        layout.addWidget(display_box)

        # Component selector combobox (will be inserted into the histogram dock widget)
        self.histogram_component_combobox = QComboBox()
        self.histogram_component_combobox.setToolTip(
            "Select which component's fraction data to display in the histogram."
        )
        self.histogram_component_combobox.currentIndexChanged.connect(
            self._on_histogram_component_changed
        )

        # Mirror of the selector shown in the statistics dock so the component
        # can be changed there too. Kept in sync with the histogram combobox.
        self.stats_component_combobox = QComboBox()
        self.stats_component_combobox.setToolTip(
            "Select which component's fraction data to display in the "
            "histogram and statistics."
        )
        self.stats_component_combobox.currentIndexChanged.connect(
            self._on_stats_component_changed
        )

        # NOTE: The widget is created here but NOT added to this tab's layout.
        # PlotterWidget wraps it in a HistogramDockWidget and docks it separately.
        self.histogram_widget = HistogramWidget(
            xlabel="Fraction",
            ylabel="Pixel count",
            bins=150,
            default_colormap_name="jet",
            range_slider_enabled=True,
            range_label_prefix="Fraction range",
            range_factor=1000,
            viewer=self.viewer,
            parent=self,
        )
        self.histogram_widget.rangeChanged.connect(
            self._on_fraction_range_changed
        )

        layout.addStretch()
        self.setLayout(root_layout)

        # Update component visibility and button states
        self._update_component_visibility()
        self._update_button_states()

    def _add_component_ui(self, idx):
        """Add UI elements for a component."""
        # Single component layout with all elements in one row
        comp_layout = QHBoxLayout()

        # Component number label
        number_label = QLabel(f"{idx + 1}.")
        number_label.setStyleSheet("font-weight: bold;")
        number_label.setMinimumWidth(20)
        comp_layout.addWidget(number_label)

        # Component name
        name_edit = QLineEdit()
        name_edit.setPlaceholderText("Component name (optional)")
        name_edit.setMaximumWidth(150)
        name_edit.setToolTip("Enter a name for this component (optional).")
        comp_layout.addWidget(name_edit)

        # Select button
        select_button = QPushButton("Select")
        select_button.setMaximumWidth(70)
        select_button.setToolTip(
            "Click here to choose how to select the component location (on plot, from cursor, or auto intersect)."
        )
        comp_layout.addWidget(select_button)

        # Create dynamic menu for the Select button
        menu = QMenu(self)
        menu.aboutToShow.connect(
            lambda idx=idx, m=menu: self._populate_select_menu(idx, m)
        )
        select_button.setMenu(menu)

        # G coordinate
        comp_layout.addWidget(QLabel("G:"))
        g_edit = QLineEdit()
        g_edit.setPlaceholderText("Real coordinate")
        g_edit.setMaximumWidth(100)
        g_edit.setToolTip("Edit the G (real) coordinate of the component.")
        comp_layout.addWidget(g_edit)

        # S coordinate
        comp_layout.addWidget(QLabel("S:"))
        s_edit = QLineEdit()
        s_edit.setPlaceholderText("Imaginary coordinate")
        s_edit.setMaximumWidth(100)
        s_edit.setToolTip(
            "Edit the S (imaginary) coordinate of the component."
        )
        comp_layout.addWidget(s_edit)

        # Lifetime input
        lifetime_label = QLabel("τ:")
        lifetime_edit = QLineEdit()
        lifetime_edit.setPlaceholderText("Lifetime (ns)")
        lifetime_edit.setMaximumWidth(80)
        lifetime_edit.setToolTip("Edit the lifetime (in ns) of the component.")
        comp_layout.addWidget(lifetime_label)
        comp_layout.addWidget(lifetime_edit)

        # Add stretch to push everything to the left
        comp_layout.addStretch()

        # Add layout to components section
        self.components_layout.addLayout(comp_layout)

        # Create component state
        comp = ComponentState(
            idx=idx,
            name_edit=name_edit,
            lifetime_edit=lifetime_edit,
            g_edit=g_edit,
            s_edit=s_edit,
            select_button=select_button,
            number_label=number_label,
            label=f"Component {idx+1}",
            text_offset=(0.02, 0.02),
        )

        # Extend components list if needed
        while len(self.components) <= idx:
            self.components.append(None)
        self.components[idx] = comp

        # Connect signals
        name_edit.textChanged.connect(
            lambda: self._on_component_name_changed(idx)
        )
        g_edit.editingFinished.connect(
            lambda: self._on_component_coords_changed(idx)
        )
        g_edit.textChanged.connect(
            lambda: self._update_component_input_styling(idx)
        )
        g_edit.textChanged.connect(
            lambda _=None: self._refresh_run_button_if_ready()
        )
        s_edit.editingFinished.connect(
            lambda: self._on_component_coords_changed(idx)
        )
        s_edit.textChanged.connect(
            lambda: self._update_component_input_styling(idx)
        )
        s_edit.textChanged.connect(
            lambda _=None: self._refresh_run_button_if_ready()
        )
        lifetime_edit.editingFinished.connect(
            lambda: self._update_component_from_lifetime(idx)
        )
        # Select button action is handled by the dynamic QMenu

        comp.ui_elements = {
            'comp_layout': comp_layout,
            'lifetime_label': lifetime_label,
        }

    def _auto_place_second_component(self):
        """Auto-place Component 2 on the universal circle based on Component 1 and the data center."""
        self._auto_place_component_by_index(1)

    def _auto_place_component_by_index(self, idx):
        """Auto-place Component idx on the universal circle based on the previous component and the data center."""
        if idx <= 0 or idx >= len(self.components):
            return

        comp_prev = self.components[idx - 1]
        comp_curr = self.components[idx]
        if comp_prev is None or comp_curr is None:
            return

        # Get previous component's position
        try:
            prev_real = float(comp_prev.g_edit.text())
            prev_imag = float(comp_prev.s_edit.text())
        except ValueError:
            show_warning(
                f"Component {idx} must be set first to auto-place Component {idx + 1}."
            )
            return

        if self.parent_widget is None:
            return

        # Get frequency
        try:
            frequency = self.parent_widget._get_frequency_from_layer()
            if frequency is None:
                show_warning(
                    f"Frequency must be set to auto-place Component {idx + 1}."
                )
                return
            current_harmonic = getattr(self.parent_widget, 'harmonic', 1)
            frequency = frequency * current_harmonic
        except (AttributeError, TypeError):
            show_warning(
                f"Frequency must be set to auto-place Component {idx + 1}."
            )
            return

        # Get data for center calculation
        layer_name = self.parent_widget.get_primary_layer_name()
        if not layer_name or layer_name not in self.viewer.layers:
            show_warning("No active layer found.")
            return

        layer = self.viewer.layers[layer_name]
        g_array = layer.metadata.get("G")
        s_array = layer.metadata.get("S")
        harmonics = layer.metadata.get("harmonics")
        mean = layer.metadata.get("original_mean")

        if g_array is None or s_array is None or harmonics is None:
            show_warning(
                "Active layer must have phasor data (G, S, harmonics) to calculate the center."
            )
            return

        harmonics = np.atleast_1d(harmonics)
        harmonic_idx = np.where(harmonics == current_harmonic)[0]
        if len(harmonic_idx) == 0:
            show_warning(
                f"Harmonic {current_harmonic} not found in the layer data."
            )
            return
        harmonic_idx = harmonic_idx[0]

        if g_array.ndim > layer.data.ndim:
            real = g_array[harmonic_idx]
            imag = s_array[harmonic_idx]
        else:
            real = g_array
            imag = s_array

        try:
            _, center_real, center_imag = phasor_center(mean, real, imag)
        except (ValueError, TypeError) as e:
            show_error(f"Error calculating phasor center: {e}")
            return

        # Intersect with the semicircle
        try:
            _, _, bound_real, bound_imag = phasor_semicircle_intersect(
                prev_real, prev_imag, center_real, center_imag
            )

            # The intersection returns arrays, we extract the float
            if np.ndim(bound_real) > 0:
                bound_real = float(np.array(bound_real).ravel()[0])
            if np.ndim(bound_imag) > 0:
                bound_imag = float(np.array(bound_imag).ravel()[0])

        except (ValueError, TypeError) as e:
            show_error(f"Error calculating intersection: {e}")
            return

        # Calculate lifetime
        try:
            lifetime_bound = phasor_to_normal_lifetime(
                bound_real, bound_imag, frequency=frequency
            )
            if np.ndim(lifetime_bound) > 0:
                lifetime_bound = float(np.array(lifetime_bound).ravel()[0])
        except (ValueError, TypeError):
            lifetime_bound = None

        # Set Component idx's position
        self._updating_from_lifetime = True
        try:
            comp_curr.g_edit.setText(f"{bound_real:.3f}")
            comp_curr.s_edit.setText(f"{bound_imag:.3f}")

            if (
                lifetime_bound is not None
                and comp_curr.lifetime_edit is not None
            ):
                comp_curr.lifetime_edit.setText(f"{lifetime_bound:.3f}")

            self._on_component_coords_changed(idx)
            self._update_component_lifetime(
                idx, current_harmonic, lifetime_bound
            )
        finally:
            self._updating_from_lifetime = False

    def _select_from_phasor_center(self, idx):
        """Open a dialog to select layer(s) and compute their pooled phasor center."""
        if self.parent_widget is None:
            return

        available_layers = [
            layer.name
            for layer in self.viewer.layers
            if isinstance(layer, Image)
            and "G" in layer.metadata
            and "S" in layer.metadata
            and "G_original" in layer.metadata
            and "S_original" in layer.metadata
        ]

        if not available_layers:
            show_warning("No active layers with phasor data found.")
            return

        comp = self.components[idx]
        preselected = [
            name
            for name in (comp.phasor_center_layers or [])
            if name in available_layers
        ]

        dialog = PhasorCenterSelectionDialog(
            available_layers, parent=self, preselected=preselected
        )
        if dialog.exec() == QDialog.Accepted:
            selected_layer_names = dialog.get_selected_layers()
            if not selected_layer_names:
                show_warning("No layers selected.")
                return

            # Remember the selection for this component so reopening the
            # dialog restores it.
            comp.phasor_center_layers = list(selected_layer_names)
            self._update_component_phasor_center_layers(
                idx, comp.phasor_center_layers
            )

            pooled_mean_list = []
            pooled_g_list = []
            pooled_s_list = []

            for name in selected_layer_names:
                layer = self.viewer.layers[name]
                samples = self.parent_widget._get_layer_phasor_samples(layer)
                if samples is not None:
                    mean_flat, g_flat, s_flat = samples
                    pooled_mean_list.append(mean_flat)
                    pooled_g_list.append(g_flat)
                    pooled_s_list.append(s_flat)

            if not pooled_mean_list:
                show_warning(
                    "Failed to retrieve phasor samples from selected layers."
                )
                return

            pooled_mean = np.concatenate(pooled_mean_list)
            pooled_g = np.concatenate(pooled_g_list)
            pooled_s = np.concatenate(pooled_s_list)

            center = self.parent_widget._compute_center_from_samples(
                pooled_mean, pooled_g, pooled_s
            )
            if center is not None:
                g_c, s_c = center
                self._set_component_coords_from_menu(idx, g_c, s_c)
            else:
                show_error(
                    "Failed to compute phasor center for selected layer(s)."
                )

    def _add_component(self):
        """Add a new component (up to maximum allowed based on harmonics)."""
        total_count = len([c for c in self.components if c is not None])
        max_components = self._get_max_components()

        if total_count >= max_components:
            self._update_button_states()
            return

        self._add_component_ui(total_count)
        self._update_component_visibility()
        self._update_analysis_options()
        self._update_button_states()

        if self.parent_widget is not None:
            self._update_lifetime_inputs_visibility()

    def _remove_component(self):
        """Remove the last component."""
        if len(self.components) <= 2:
            self._update_button_states()
            return

        last_idx = len(self.components) - 1
        comp = self.components[last_idx]

        if comp is not None:
            if comp.dot is not None:
                comp.dot.remove()
            if comp.text is not None:
                comp.text.remove()

            if hasattr(comp, 'ui_elements'):
                comp_layout = comp.ui_elements['comp_layout']
                while comp_layout.count():
                    item = comp_layout.takeAt(0)
                    if item.widget():
                        item.widget().deleteLater()
                self.components_layout.removeItem(comp_layout)

        self.components.pop()

        self._update_component_numbering()

        if len(self.components) == 2:
            self._update_analysis_options()

            index = self.analysis_type_combo.findText("Linear Projection")
            if index >= 0:
                self.analysis_type_combo.setCurrentIndex(index)

        if self.component_line is not None:
            with contextlib.suppress(ValueError, AttributeError):
                self.component_line.remove()
            self.component_line = None

        if self.component_polygon is not None:
            with contextlib.suppress(ValueError, AttributeError):
                self.component_polygon.remove()
            self.component_polygon = None

        self._remove_histogram_overlay()

        self._update_component_visibility()
        self._update_analysis_options()
        self._update_button_states()

        if self._analysis_attempted:
            self._update_all_component_styling()

        self.draw_line_between_components()

        if self.parent_widget is not None and hasattr(
            self.parent_widget, '_get_frequency_from_layer'
        ):
            self._update_lifetime_inputs_visibility()

        if self.parent_widget is not None:
            self.parent_widget.canvas_widget.canvas.draw_idle()

        self._remove_last_component_from_settings()

    def _remove_last_component_from_settings(self):
        """Remove the last component from the settings in metadata."""
        if self._updating_settings:
            return

        if not self.current_image_layer_name:
            return

        # Check if layer still exists (defensive check for cleanup/teardown)
        if self.current_image_layer_name not in self.viewer.layers:
            return

        layer = self.viewer.layers[self.current_image_layer_name]
        if (
            'settings' not in layer.metadata
            or 'component_analysis' not in layer.metadata['settings']
        ):
            return

        settings = layer.metadata['settings']['component_analysis']

        if 'components' in settings and len(settings['components']) > 0:
            max_idx_str = max(settings['components'].keys(), key=int)
            del settings['components'][max_idx_str]

    def _get_max_components(self):
        """Get maximum number of components based on available harmonics."""
        if self.parent_widget is None:
            return 3

        try:
            available_harmonics = self._get_available_harmonics()
            num_harmonics = len(available_harmonics)
            if num_harmonics == 0:
                return 3

            return 2 * num_harmonics + 1
        except Exception:  # noqa: BLE001
            return 3

    def _components_validation(self):
        """Return ``None`` if the analysis can run, else the missing-input msg."""
        if not self.parent_widget.get_selected_layers():
            return "Select at least one image layer with phasor features."
        comps = [c for c in self.components if c is not None]
        if not comps:
            return "Add at least two components."
        for comp in comps:
            g_text = comp.g_edit.text().strip() if comp.g_edit else ""
            s_text = comp.s_edit.text().strip() if comp.s_edit else ""
            if not g_text or not s_text:
                return "Enter G and S coordinates for every component."
            try:
                float(g_text)
                float(s_text)
            except ValueError:
                return "Component coordinates must be valid numbers."
        return None

    def _refresh_run_button_if_ready(self):
        """Re-evaluate the Run button state if it has been wired up."""
        refresh = getattr(self, '_refresh_run_button', None)
        if refresh is not None:
            refresh()

    def _update_button_states(self):
        """Update the enabled/disabled state of add/remove component buttons."""
        total_count = len([c for c in self.components if c is not None])
        max_components = self._get_max_components()

        self.add_component_btn.setEnabled(total_count < max_components)

        self.remove_component_btn.setEnabled(total_count > 2)

        self._refresh_run_button_if_ready()

    def _clear_components(self):
        """Clear all component visualizations and input fields."""
        for comp in self.components:
            if comp.dot is not None:
                with contextlib.suppress(ValueError, AttributeError):
                    comp.dot.remove()
                comp.dot = None

            if comp.text is not None:
                with contextlib.suppress(ValueError, AttributeError):
                    comp.text.remove()
                comp.text = None

            comp.g_edit.clear()
            comp.s_edit.clear()
            comp.name_edit.clear()
            if comp.lifetime_edit is not None:
                comp.lifetime_edit.clear()

        if self.component_line is not None:
            with contextlib.suppress(ValueError, AttributeError):
                self.component_line.remove()
            self.component_line = None

        if self.component_polygon is not None:
            with contextlib.suppress(ValueError, AttributeError):
                self.component_polygon.remove()
            self.component_polygon = None

        self._remove_histogram_overlay()

        self._update_components_setting_in_metadata('components', {})

        if self.parent_widget is not None:
            self.parent_widget.canvas_widget.canvas.draw_idle()

    def _update_component_numbering(self):
        """Update the numbering labels for all components."""
        for i, comp in enumerate(self.components):
            if comp is not None and comp.number_label is not None:
                comp.number_label.setText(f"{i + 1}.")

    def _get_default_components_settings(self):
        """Get default settings dictionary for components parameters."""
        return {
            'analysis_type': 'Linear Projection',
            'components': {},
            # NOTE: keys here must match the ones written by
            # ``_on_plot_setting_changed`` / ``_pick_label_color`` and read by
            # the restore methods, otherwise edits are persisted under one key
            # but restored from another (and never re-applied).
            'two_component_line_settings': {
                'show_colormap_line': True,
                'show_component_dots': True,
                'line_offset': 0.0,
                'line_width': 3.0,
                'line_alpha': 1.0,
                'default_component_color': 'dimgray',
                'show_fraction_histogram': False,
                'histogram_overlay_height': 0.3,
                'histogram_offset': 0.0,
                'histogram_alpha': 0.75,
            },
            'two_components_label_settings': {
                'fontsize': 10,
                'bold': False,
                'italic': False,
                'color': 'black',
            },
        }

    def _restore_line_and_label_settings(self, settings):
        """Restore the two-component line / histogram-overlay and label styles.

        User edits are persisted under ``two_component_line_settings`` and
        ``two_components_label_settings`` (see ``_on_plot_setting_changed`` and
        ``_pick_label_color``). Older metadata used ``line_settings`` /
        ``label_settings``, so fall back to those for backward compatibility.
        """
        line_settings = (
            settings.get('two_component_line_settings')
            or settings.get('line_settings')
            or {}
        )
        if line_settings:
            self.show_colormap_line = line_settings.get(
                'show_colormap_line', True
            )
            self.show_component_dots = line_settings.get(
                'show_component_dots', True
            )
            self.line_offset = line_settings.get('line_offset', 0.0)
            self.line_width = line_settings.get('line_width', 3.0)
            self.line_alpha = line_settings.get('line_alpha', 1.0)
            self.default_component_color = line_settings.get(
                'default_component_color', 'dimgray'
            )
            self.show_fraction_histogram = line_settings.get(
                'show_fraction_histogram', False
            )
            self.histogram_overlay_height = line_settings.get(
                'histogram_overlay_height', 0.3
            )
            self.histogram_offset = line_settings.get('histogram_offset', 0.0)
            self.histogram_alpha = line_settings.get('histogram_alpha', 0.75)

        label_settings = (
            settings.get('two_components_label_settings')
            or settings.get('label_settings')
            or {}
        )
        if label_settings:
            self.label_fontsize = label_settings.get('fontsize', 10)
            self.label_bold = label_settings.get('bold', False)
            self.label_italic = label_settings.get('italic', False)
            self.label_color = label_settings.get('color', 'black')

    def _update_components_setting_in_metadata(self, key_path, value):
        """Update a specific component setting in the current layer's metadata."""
        layer_name = self.parent_widget.get_primary_layer_name()
        if not layer_name:
            return

        layer = self.viewer.layers[layer_name]

        if 'settings' not in layer.metadata:
            layer.metadata['settings'] = {}
        if 'component_analysis' not in layer.metadata['settings']:
            layer.metadata['settings']['component_analysis'] = {}

        keys = key_path.split('.')
        settings = layer.metadata['settings']['component_analysis']
        for key in keys[:-1]:
            if key not in settings:
                settings[key] = {}
            settings = settings[key]
        settings[keys[-1]] = value

    def _restore_components_ui_only_from_metadata(self):
        """Restore component UI values and visual dots from metadata without running analysis.

        This restores input fields (names, G/S coordinates, lifetimes), component
        dots on the plot, and draws lines between components, but does NOT run
        the analysis (no fraction layers are created). The user must click
        the run button to execute the analysis.
        """

        layer_name = self.parent_widget.get_primary_layer_name()

        if not layer_name:
            return

        layer = self.viewer.layers[layer_name]

        if 'settings' not in layer.metadata:
            return

        if 'component_analysis' not in layer.metadata['settings']:
            return

        self._updating_settings = True
        try:
            settings = layer.metadata['settings']['component_analysis']

            self._clear_components_display()

            if 'analysis_type' in settings:
                self.analysis_type = settings['analysis_type']
                if hasattr(self, 'analysis_type_combo'):
                    index = self.analysis_type_combo.findText(
                        self.analysis_type
                    )
                    if index >= 0:
                        self.analysis_type_combo.setCurrentIndex(index)

            if 'last_analysis_harmonic' in settings:
                last_harmonic = settings['last_analysis_harmonic']
                if hasattr(self.parent_widget, 'harmonic_spinbox'):
                    self.parent_widget.harmonic_spinbox.setValue(last_harmonic)
                current_harmonic = last_harmonic
            else:
                current_harmonic = getattr(self.parent_widget, 'harmonic', 1)

            current_harmonic_key = str(current_harmonic)

            if 'components' in settings and isinstance(
                settings['components'], dict
            ):
                if settings['components']:
                    max_idx = max(int(k) for k in settings['components'])
                else:
                    max_idx = -1

                while len(self.components) < max_idx + 1:
                    self._add_component_ui(len(self.components))

                while (
                    len(self.components) > max_idx + 1
                    and len(self.components) > 2
                ):
                    last_idx = len(self.components) - 1
                    comp = self.components[last_idx]

                    if comp is not None:
                        if comp.dot is not None:
                            comp.dot.remove()
                            comp.dot = None
                        if comp.text is not None:
                            comp.text.remove()
                            comp.text = None

                        if hasattr(comp, 'ui_elements'):
                            comp_layout = comp.ui_elements['comp_layout']
                            while comp_layout.count():
                                item = comp_layout.takeAt(0)
                                if item.widget():
                                    item.widget().deleteLater()
                            self.components_layout.removeItem(comp_layout)

                    self.components.pop()

                for idx_str, comp_data in settings['components'].items():
                    idx = int(idx_str)

                    if (
                        idx >= len(self.components)
                        or self.components[idx] is None
                    ):
                        continue

                    comp = self.components[idx]
                    name = comp_data.get('name', '')

                    if name:
                        comp.name_edit.setText(name)

                    comp.phasor_center_layers = list(
                        comp_data.get('phasor_center_layers', [])
                    )

                    gs_harmonics = comp_data.get('gs_harmonics', {})

                    if current_harmonic_key in gs_harmonics:
                        harmonic_data = gs_harmonics[current_harmonic_key]

                        g = harmonic_data.get('g')
                        s = harmonic_data.get('s')
                        lifetime = harmonic_data.get('lifetime')

                        if g is not None:
                            comp.g_edit.setText(f"{g:.3f}")
                        if s is not None:
                            comp.s_edit.setText(f"{s:.3f}")
                        if (
                            lifetime is not None
                            and comp.lifetime_edit is not None
                        ):
                            comp.lifetime_edit.setText(str(lifetime))

                        if g is not None and s is not None:
                            self._create_component_at_coordinates(idx, g, s)

            self._restore_line_and_label_settings(settings)

            # Draw visual elements (lines between components) but do NOT
            # run analysis or create fraction layers.
            components_created = [
                c
                for c in self.components
                if c is not None and c.dot is not None
            ]

            if len(components_created) >= 2:
                self.draw_line_between_components()

            self._update_button_states()

            self._update_component_visibility()

        except Exception as e:  # noqa: BLE001
            show_error(
                f"Error restoring component settings from metadata: {str(e)}"
            )
        finally:
            self._updating_settings = False

    def _restore_and_recreate_components_from_metadata(self):
        """Restore all components settings, recreate visual elements, and run analysis from metadata."""

        layer_name = self.parent_widget.get_primary_layer_name()

        if not layer_name:
            return

        layer = self.viewer.layers[layer_name]

        if 'settings' not in layer.metadata:
            return

        if 'component_analysis' not in layer.metadata['settings']:
            return

        self._updating_settings = True
        try:
            settings = layer.metadata['settings']['component_analysis']

            self._clear_components_display()

            if 'analysis_type' in settings:
                self.analysis_type = settings['analysis_type']
                if hasattr(self, 'analysis_type_combo'):
                    index = self.analysis_type_combo.findText(
                        self.analysis_type
                    )
                    if index >= 0:
                        self.analysis_type_combo.setCurrentIndex(index)

            if 'last_analysis_harmonic' in settings:
                last_harmonic = settings['last_analysis_harmonic']
                if hasattr(self.parent_widget, 'harmonic_spinbox'):
                    self.parent_widget.harmonic_spinbox.setValue(last_harmonic)
                current_harmonic = last_harmonic
            else:
                current_harmonic = getattr(self.parent_widget, 'harmonic', 1)

            current_harmonic_key = str(current_harmonic)

            if 'components' in settings and isinstance(
                settings['components'], dict
            ):
                if settings['components']:
                    max_idx = max(int(k) for k in settings['components'])
                else:
                    max_idx = -1

                while len(self.components) < max_idx + 1:
                    self._add_component_ui(len(self.components))

                while (
                    len(self.components) > max_idx + 1
                    and len(self.components) > 2
                ):
                    last_idx = len(self.components) - 1
                    comp = self.components[last_idx]

                    if comp is not None:
                        if comp.dot is not None:
                            comp.dot.remove()
                            comp.dot = None
                        if comp.text is not None:
                            comp.text.remove()
                            comp.text = None

                        if hasattr(comp, 'ui_elements'):
                            comp_layout = comp.ui_elements['comp_layout']
                            while comp_layout.count():
                                item = comp_layout.takeAt(0)
                                if item.widget():
                                    item.widget().deleteLater()
                            self.components_layout.removeItem(comp_layout)

                    self.components.pop()

                for idx_str, comp_data in settings['components'].items():
                    idx = int(idx_str)

                    if (
                        idx >= len(self.components)
                        or self.components[idx] is None
                    ):
                        continue

                    comp = self.components[idx]
                    name = comp_data.get('name', '')

                    if name:
                        comp.name_edit.setText(name)

                    comp.phasor_center_layers = list(
                        comp_data.get('phasor_center_layers', [])
                    )

                    gs_harmonics = comp_data.get('gs_harmonics', {})

                    if current_harmonic_key in gs_harmonics:
                        harmonic_data = gs_harmonics[current_harmonic_key]

                        g = harmonic_data.get('g')
                        s = harmonic_data.get('s')
                        lifetime = harmonic_data.get('lifetime')

                        if g is not None:
                            comp.g_edit.setText(f"{g:.3f}")
                        if s is not None:
                            comp.s_edit.setText(f"{s:.3f}")
                        if (
                            lifetime is not None
                            and comp.lifetime_edit is not None
                        ):
                            comp.lifetime_edit.setText(str(lifetime))

                        if g is not None and s is not None:
                            self._create_component_at_coordinates(idx, g, s)

            self._restore_line_and_label_settings(settings)

            components_created = [
                c
                for c in self.components
                if c is not None and c.dot is not None
            ]

            if len(components_created) >= 2:
                analysis_type = settings.get(
                    'analysis_type', 'Linear Projection'
                )

                if (
                    analysis_type == 'Linear Projection'
                    and len(components_created) == 2
                ) or (
                    analysis_type == 'Component Fit'
                    and len(components_created) >= 2
                ):

                    if (
                        analysis_type == 'Component Fit'
                        and len(components_created) > 2
                    ):
                        required_harmonics = self._get_required_harmonics(
                            len(components_created)
                        )

                        harmonics_in_metadata = set()
                        for comp_data in settings['components'].values():
                            gs_harmonics = comp_data.get('gs_harmonics', {})
                            for harmonic_key in gs_harmonics:
                                harmonics_in_metadata.add(int(harmonic_key))

                        if len(harmonics_in_metadata) >= required_harmonics:
                            self._run_analysis()
                            self._restore_fraction_layer_colormaps(
                                settings, current_harmonic_key
                            )

                    else:
                        self._run_analysis()
                        self._restore_fraction_layer_colormaps(
                            settings, current_harmonic_key
                        )

            else:
                if len(components_created) >= 2:
                    self.draw_line_between_components()

            self._update_button_states()

            self._update_component_visibility()

        except Exception as e:  # noqa: BLE001
            show_error(
                f"Error restoring component settings from metadata: {str(e)}"
            )
        finally:
            self._updating_settings = False

    def _restore_fraction_layer_colormaps(self, settings, harmonic_key):
        """Restore colormap settings for fraction layers after analysis."""
        if not settings.get('components'):
            return

        for idx_str, comp_data in settings['components'].items():
            idx = int(idx_str)

            gs_harmonics = comp_data.get('gs_harmonics', {})
            if harmonic_key not in gs_harmonics:
                continue

            harmonic_data = gs_harmonics[harmonic_key]

            colormap_name = harmonic_data.get('colormap_name')
            colormap_colors = harmonic_data.get('colormap_colors')
            contrast_limits = harmonic_data.get('contrast_limits')
            gamma = harmonic_data.get('gamma')

            if not (colormap_name or colormap_colors):
                continue

            comp_name = comp_data.get('name') or f"Component {idx + 1}"

            possible_layer_names = [
                f"{comp_name} fractions: {self.current_image_layer_name}",  # Linear projection
                f"{comp_name} fraction: {self.current_image_layer_name}",  # Component fit
            ]

            fraction_layer = None
            for layer_name in possible_layer_names:
                if layer_name in self.viewer.layers:
                    fraction_layer = self.viewer.layers[layer_name]
                    break

            if fraction_layer is None:
                continue

            with contextlib.suppress(Exception):
                fraction_layer.events.colormap.disconnect(
                    self._on_colormap_changed
                )
                fraction_layer.events.gamma.disconnect(
                    self._on_colormap_changed
                )

            try:
                if colormap_colors is not None:
                    colors = (
                        np.array(colormap_colors)
                        if isinstance(colormap_colors, list)
                        else colormap_colors
                    )
                    from napari.utils.colormaps import Colormap

                    custom_colormap = Colormap(
                        colors=colors, name=colormap_name or "custom"
                    )
                    fraction_layer.colormap = custom_colormap
                elif colormap_name:
                    fraction_layer.colormap = colormap_name

                if contrast_limits:
                    fraction_layer.contrast_limits = tuple(contrast_limits)

                if gamma is not None:
                    fraction_layer.gamma = gamma

            finally:
                fraction_layer.events.colormap.connect(
                    self._on_colormap_changed
                )
                fraction_layer.events.gamma.connect(self._on_colormap_changed)

        if (
            len(settings.get('components', {})) == 2
            and self.analysis_type == "Linear Projection"
        ) and self.comp1_fractions_layer is not None:
            self.fractions_colormap = (
                self.comp1_fractions_layer.colormap.colors
            )
            self.colormap_contrast_limits = (
                self.comp1_fractions_layer.contrast_limits
            )
            self.fractions_gamma = self.comp1_fractions_layer.gamma

        self._update_component_colors()
        self.draw_line_between_components()

        if self.parent_widget is not None:
            self.parent_widget.canvas_widget.canvas.draw_idle()

    def _make_slider_spin_row(
        self,
        label,
        minv,
        maxv,
        value,
        decimals,
        on_value,
        *,
        center_fill=False,
        tooltip=None,
        step=None,
    ):
        """Build a labelled slider + spinbox pair kept in sync.

        ``on_value`` is invoked with the float value whenever either the slider
        or the spinbox changes. The value can therefore be dragged or typed
        directly. Returns ``(row_layout, slider, spin)``.
        """
        factor = 10**decimals
        row = QHBoxLayout()
        lbl = QLabel(label)
        row.addWidget(lbl)

        slider = (
            CenterFillSlider(Qt.Horizontal)
            if center_fill
            else QSlider(Qt.Horizontal)
        )
        slider.setRange(int(round(minv * factor)), int(round(maxv * factor)))
        slider.setValue(int(round(value * factor)))

        spin = QDoubleSpinBox()
        spin.setDecimals(decimals)
        spin.setRange(minv, maxv)
        spin.setSingleStep(step if step is not None else 1.0 / factor)
        spin.setValue(value)
        spin.setMaximumWidth(85)

        if tooltip:
            for w in (lbl, slider, spin):
                w.setToolTip(tooltip)

        def on_slider(iv):
            if self._syncing_slider_spin:
                return
            self._syncing_slider_spin = True
            try:
                spin.setValue(iv / factor)
            finally:
                self._syncing_slider_spin = False
            on_value(iv / factor)

        def on_spin(val):
            if self._syncing_slider_spin:
                return
            self._syncing_slider_spin = True
            try:
                slider.setValue(int(round(val * factor)))
            finally:
                self._syncing_slider_spin = False
            on_value(val)

        slider.valueChanged.connect(on_slider)
        spin.valueChanged.connect(on_spin)

        row.addWidget(slider)
        row.addWidget(spin)
        return row, slider, spin

    def _open_plot_settings_dialog(self):
        """Open dialog to edit line and fraction-histogram overlay settings."""
        if self.plot_dialog is not None and self.plot_dialog.isVisible():
            self.plot_dialog.raise_()
            self.plot_dialog.activateWindow()
            return

        self.plot_dialog = QDialog(self)
        self.plot_dialog.setWindowTitle("Component Line & Histogram Settings")
        self.plot_dialog.setMinimumWidth(460)
        vbox = QVBoxLayout(self.plot_dialog)

        # Top row: display checkboxes side by side.
        checks_row = QHBoxLayout()
        self.show_dots_checkbox = QCheckBox("Show component positions")
        self.show_dots_checkbox.setChecked(self.show_component_dots)
        self.show_dots_checkbox.stateChanged.connect(
            self._on_plot_setting_changed
        )
        checks_row.addWidget(self.show_dots_checkbox)
        checks_row.addStretch()
        vbox.addLayout(checks_row)

        # --- Line group -----------------------------------------------------
        line_group = QGroupBox("Line")
        line_layout = QVBoxLayout(line_group)

        self.colormap_line_checkbox = QCheckBox("Overlay colormap")
        self.colormap_line_checkbox.setChecked(self.show_colormap_line)
        self.colormap_line_checkbox.stateChanged.connect(
            self._on_plot_setting_changed
        )
        line_layout.addWidget(self.colormap_line_checkbox)

        width_row, self.line_width_slider, self.line_width_spin = (
            self._make_slider_spin_row(
                "Width:",
                0.5,
                20.0,
                self.line_width,
                1,
                self._on_line_width_changed,
                step=0.5,
            )
        )
        (
            line_transp_row,
            self.line_transparency_slider,
            self.line_transparency_spin,
        ) = self._make_slider_spin_row(
            "Transparency:",
            0.0,
            1.0,
            1.0 - self.line_alpha,
            2,
            self._on_line_transparency_changed,
        )
        top_line_row = QHBoxLayout()
        top_line_row.addLayout(width_row)
        top_line_row.addSpacing(12)
        top_line_row.addLayout(line_transp_row)
        line_layout.addLayout(top_line_row)

        offset_row, self.line_offset_slider, self.line_offset_spin = (
            self._make_slider_spin_row(
                "Offset:",
                -1.0,
                1.0,
                self.line_offset,
                3,
                self._on_line_offset_changed,
                center_fill=True,
                tooltip="Shift the line perpendicular to its direction.",
            )
        )
        color_row = QHBoxLayout()
        color_row.addWidget(QLabel("Default color:"))
        self.color_button = QPushButton()
        self.color_button.setMaximumWidth(80)
        self.color_button.setStyleSheet(
            f"background-color: {self.default_component_color}; "
            "border: 1px solid black;"
        )
        self.color_button.clicked.connect(self._on_color_button_clicked)
        color_row.addWidget(self.color_button)
        color_row.addStretch()

        bottom_line_row = QHBoxLayout()
        bottom_line_row.addLayout(offset_row)
        bottom_line_row.addSpacing(12)
        bottom_line_row.addLayout(color_row)
        line_layout.addLayout(bottom_line_row)
        vbox.addWidget(line_group)

        # --- Fraction histogram group --------------------------------------
        hist_group = QGroupBox("Fraction histogram overlay")
        hist_layout = QVBoxLayout(hist_group)

        self.fraction_histogram_checkbox = QCheckBox(
            "Overlay fraction histogram"
        )
        self.fraction_histogram_checkbox.setChecked(
            self.show_fraction_histogram
        )
        self.fraction_histogram_checkbox.setToolTip(
            "Overlay the histogram of the first component's fraction on top "
            "of the line joining the components, colored with the line's "
            "colormap. Available for a two-component Linear Projection."
        )
        self.fraction_histogram_checkbox.stateChanged.connect(
            self._on_plot_setting_changed
        )
        hist_layout.addWidget(self.fraction_histogram_checkbox)

        (
            height_row,
            self.histogram_height_slider,
            self.histogram_height_spin,
        ) = self._make_slider_spin_row(
            "Height:",
            0.05,
            1.0,
            self.histogram_overlay_height,
            2,
            self._on_histogram_height_changed,
        )
        (
            hist_transp_row,
            self.histogram_transparency_slider,
            self.histogram_transparency_spin,
        ) = self._make_slider_spin_row(
            "Transparency:",
            0.0,
            1.0,
            1.0 - self.histogram_alpha,
            2,
            self._on_histogram_transparency_changed,
        )
        top_hist_row = QHBoxLayout()
        top_hist_row.addLayout(height_row)
        top_hist_row.addSpacing(12)
        top_hist_row.addLayout(hist_transp_row)
        hist_layout.addLayout(top_hist_row)

        (
            hist_offset_row,
            self.histogram_offset_slider,
            self.histogram_offset_spin,
        ) = self._make_slider_spin_row(
            "Offset:",
            -1.0,
            1.0,
            self.histogram_offset,
            3,
            self._on_histogram_offset_changed,
            center_fill=True,
            tooltip="Shift the histogram relative to the line. Positive keeps "
            "it on one side; negative flips it to the other side. The "
            "magnitude is the distance from the line.",
        )
        hist_layout.addLayout(hist_offset_row)
        vbox.addWidget(hist_group)

        # Buttons
        buttons_layout = QHBoxLayout()
        reset_button = QPushButton("Reset")
        reset_button.clicked.connect(self._reset_plot_settings)
        buttons_layout.addWidget(reset_button)
        buttons_layout.addStretch()
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.plot_dialog.close)
        buttons_layout.addWidget(close_button)
        vbox.addLayout(buttons_layout)

        self.plot_dialog.show()

    def _on_color_button_clicked(self):
        """Handle color button click to open color dialog."""
        color = QColorDialog.getColor()
        if color.isValid():
            self.default_component_color = color.name()

            self.color_button.setStyleSheet(
                f"background-color: {self.default_component_color}; border: 1px solid black;"
            )

            self._update_components_setting_in_metadata(
                'two_component_line_settings.default_component_color',
                self.default_component_color,
            )
            self._on_plot_setting_changed()

    def _update_analysis_options(self):
        """Update available analysis options based on number of components."""
        num_components = len(self.components)
        current_selection = self.analysis_type_combo.currentText()

        self.analysis_type_combo.clear()

        if num_components == 2:
            self.analysis_type_combo.addItems(
                ["Linear Projection", "Component Fit"]
            )
        else:
            self.analysis_type_combo.addItems(["Component Fit"])

        index = self.analysis_type_combo.findText(current_selection)
        if index >= 0:
            self.analysis_type_combo.setCurrentIndex(index)

        self.analysis_type = self.analysis_type_combo.currentText()

    def _on_analysis_type_changed(self, analysis_type):
        """Handle analysis type change."""
        self.analysis_type = analysis_type

        if analysis_type == "Linear Projection":
            self.calculate_button.setText("Display Component Fraction Images")
        else:
            self.calculate_button.setText("Run Multi-Component Analysis")

        self.draw_line_between_components()

    def _on_harmonic_changed(self, new_harmonic):
        """Handle harmonic changes - store current components and restore for new harmonic."""
        self._clear_components_display()
        self.current_harmonic = new_harmonic

        self._refresh_components_info_label()
        self._restore_components_for_harmonic(new_harmonic)
        self._update_component_visibility()

        if self._analysis_attempted:
            self._update_all_component_styling()

        if self.parent_widget is not None:
            self._update_lifetime_inputs_visibility()

    def _refresh_components_info_label(self):
        """Update the components info label with colored harmonic text."""
        self.components_info_label.setText(
            f"Components at <span style='color: #00FFFF;'>harmonic {self.current_harmonic}</span>"
        )

    def _get_required_harmonics(self, num_components):
        """Minimum number of harmonics required for ``num_components``.

        Delegates to the shared :func:`napari_phasors._utils.
        required_component_harmonics` so the interactive tab and the batch
        analysis pipeline stay in agreement.
        """
        return required_component_harmonics(num_components)

    def _clear_components_display(self):
        """Clear component display without removing from storage."""
        for comp in self.components:
            if comp is not None:
                if comp.dot is not None:
                    comp.dot.remove()
                    comp.dot = None
                if comp.text is not None:
                    comp.text.remove()
                    comp.text = None

                comp.g_edit.clear()
                comp.s_edit.clear()

        if (
            hasattr(self, 'component_polygon')
            and self.component_polygon is not None
        ):
            with contextlib.suppress(ValueError, AttributeError):
                self.component_polygon.remove()
            self.component_polygon = None

        if self.component_line is not None:
            with contextlib.suppress(ValueError, AttributeError):
                self.component_line.remove()
            self.component_line = None

        self._remove_histogram_overlay()

        if self.parent_widget is not None:
            self.parent_widget.canvas_widget.canvas.draw_idle()

    def _restore_components_for_harmonic(self, harmonic):
        """Restore component states for a specific harmonic."""
        if not self.current_image_layer_name:
            return

        try:
            layer = self.viewer.layers[self.current_image_layer_name]
        except KeyError:
            self.current_image_layer_name = None
            return

        if (
            'settings' not in layer.metadata
            or 'component_analysis' not in layer.metadata['settings']
        ):
            return

        settings = layer.metadata['settings']['component_analysis']
        components_data = settings.get('components', {})
        harmonic_key = str(harmonic)

        components_with_data = set()

        for idx_str, comp_data in components_data.items():
            idx = int(idx_str)

            if idx >= len(self.components) or self.components[idx] is None:
                continue

            comp = self.components[idx]
            gs_harmonics = comp_data.get('gs_harmonics', {})

            if harmonic_key not in gs_harmonics:
                comp.g_edit.clear()
                comp.s_edit.clear()
                if comp.lifetime_edit is not None:
                    comp.lifetime_edit.clear()
                continue

            harmonic_data = gs_harmonics[harmonic_key]
            g = harmonic_data.get('g')
            s = harmonic_data.get('s')

            if g is None or s is None:
                comp.g_edit.clear()
                comp.s_edit.clear()
                if comp.lifetime_edit is not None:
                    comp.lifetime_edit.clear()
                continue

            components_with_data.add(idx)

            comp.g_edit.setText(f"{g:.3f}")
            comp.s_edit.setText(f"{s:.3f}")

            stored_name = comp_data.get('name', '')
            current_name = comp.name_edit.text().strip()
            if stored_name and (
                not current_name or current_name.startswith("Component ")
            ):
                comp.name_edit.setText(stored_name)

            lifetime = harmonic_data.get('lifetime')
            if lifetime is not None and comp.lifetime_edit is not None:
                comp.lifetime_edit.setText(str(lifetime))
            elif comp.lifetime_edit is not None:
                comp.lifetime_edit.clear()

            self._create_component_at_coordinates(idx, g, s)

        for comp in self.components:
            if comp is not None and comp.idx not in components_with_data:
                comp.g_edit.clear()
                comp.s_edit.clear()
                if comp.lifetime_edit is not None:
                    comp.lifetime_edit.clear()

                if comp.dot is not None:
                    comp.dot.remove()
                    comp.dot = None
                if comp.text is not None:
                    comp.text.remove()
                    comp.text = None

        self.draw_line_between_components()
        self._update_component_colors()
        if self.parent_widget is not None:
            self.parent_widget.canvas_widget.canvas.draw_idle()

    def _open_label_style_dialog(self):
        """Open dialog to edit label style settings."""
        if self.style_dialog is not None and self.style_dialog.isVisible():
            self.style_dialog.raise_()
            self.style_dialog.activateWindow()
            return

        self.style_dialog = QDialog(self)
        self.style_dialog.setWindowTitle("Component Label Style")
        vbox = QVBoxLayout(self.style_dialog)

        row = QHBoxLayout()
        row.addWidget(QLabel("Size:"))
        self.fontsize_spin = QSpinBox()
        self.fontsize_spin.setRange(6, 72)
        self.fontsize_spin.setValue(self.label_fontsize)
        self.fontsize_spin.valueChanged.connect(self._on_label_style_changed)
        row.addWidget(self.fontsize_spin)

        self.bold_checkbox = QCheckBox("Bold")
        self.bold_checkbox.setChecked(self.label_bold)
        self.bold_checkbox.stateChanged.connect(self._on_label_style_changed)
        row.addWidget(self.bold_checkbox)

        self.italic_checkbox = QCheckBox("Italic")
        self.italic_checkbox.setChecked(self.label_italic)
        self.italic_checkbox.stateChanged.connect(self._on_label_style_changed)
        row.addWidget(self.italic_checkbox)

        self.color_button = QPushButton("Color")
        self.color_button.clicked.connect(self._pick_label_color)
        row.addWidget(self.color_button)

        row.addStretch()
        vbox.addLayout(row)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.style_dialog.close)
        vbox.addWidget(buttons)

        self.style_dialog.show()

    def _reset_plot_settings(self):
        """Reset all plot settings to their default values."""
        default_show_colormap_line = True
        default_show_component_dots = True
        default_line_offset = 0.0
        default_line_width = 3.0
        default_line_alpha = 1
        default_component_color = 'dimgray'

        default_show_fraction_histogram = False
        default_histogram_overlay_height = 0.3
        default_histogram_offset = 0.0
        default_histogram_alpha = 0.75

        self.show_colormap_line = default_show_colormap_line
        self.show_component_dots = default_show_component_dots
        self.line_offset = default_line_offset
        self.line_width = default_line_width
        self.line_alpha = default_line_alpha
        self.default_component_color = default_component_color
        self.show_fraction_histogram = default_show_fraction_histogram
        self.histogram_overlay_height = default_histogram_overlay_height
        self.histogram_offset = default_histogram_offset
        self.histogram_alpha = default_histogram_alpha

        self.colormap_line_checkbox.setChecked(default_show_colormap_line)
        self.show_dots_checkbox.setChecked(default_show_component_dots)

        if hasattr(self, 'fraction_histogram_checkbox'):
            self.fraction_histogram_checkbox.setChecked(
                default_show_fraction_histogram
            )

        # Updating the spinboxes cascades to the linked sliders and their
        # value handlers, so this both refreshes the UI and re-applies state.
        for attr, val in (
            ('line_offset_spin', default_line_offset),
            ('line_width_spin', default_line_width),
            ('line_transparency_spin', 1.0 - default_line_alpha),
            ('histogram_height_spin', default_histogram_overlay_height),
            ('histogram_offset_spin', default_histogram_offset),
            ('histogram_transparency_spin', 1.0 - default_histogram_alpha),
        ):
            spin = getattr(self, attr, None)
            if spin is not None:
                spin.setValue(val)

        if hasattr(self, 'color_button'):
            self.color_button.setStyleSheet(
                f"background-color: {default_component_color}; border: 1px solid black;"
            )

        for comp in self.components:
            if comp.dot is not None:
                comp.dot.set_alpha(default_line_alpha)

        components_tab_is_active = (
            self.parent_widget is not None
            and getattr(self.parent_widget, "tab_widget", None) is not None
            and self.parent_widget.tab_widget.currentWidget()
            is self.parent_widget.components_tab
        )
        self.set_artists_visible(components_tab_is_active)

        self.draw_line_between_components()
        if self.parent_widget is not None:
            self.parent_widget.canvas_widget.canvas.draw_idle()

    def _apply_saved_colormap_settings(self):
        """Apply saved colormap settings to fraction layers if they exist."""
        if self.comp1_fractions_layer is not None and hasattr(
            self, '_saved_colormap_name'
        ):

            try:
                self.comp1_fractions_layer.events.colormap.disconnect(
                    self._on_colormap_changed
                )
                self.comp1_fractions_layer.events.contrast_limits.disconnect(
                    self._on_contrast_limits_changed
                )
                self.comp1_fractions_layer.events.gamma.disconnect(
                    self._on_colormap_changed
                )

                if self._saved_colormap_colors is not None:
                    from napari.utils.colormaps import Colormap

                    if isinstance(self._saved_colormap_colors, list):
                        saved_colors = np.array(self._saved_colormap_colors)
                    else:
                        saved_colors = self._saved_colormap_colors

                    saved_colormap = Colormap(
                        colors=saved_colors, name="saved_custom"
                    )
                    self.comp1_fractions_layer.colormap = saved_colormap
                else:
                    self.comp1_fractions_layer.colormap = (
                        self._saved_colormap_name
                    )

                if isinstance(self._saved_contrast_limits, list):
                    saved_limits = tuple(self._saved_contrast_limits)
                else:
                    saved_limits = self._saved_contrast_limits

                self.comp1_fractions_layer.contrast_limits = saved_limits

                self.fractions_colormap = (
                    self.comp1_fractions_layer.colormap.colors
                )
                self.colormap_contrast_limits = (
                    self.comp1_fractions_layer.contrast_limits
                )

                self.comp1_fractions_layer.events.colormap.connect(
                    self._on_colormap_changed
                )
                self.comp1_fractions_layer.events.contrast_limits.connect(
                    self._on_contrast_limits_changed
                )
                self.comp1_fractions_layer.events.gamma.connect(
                    self._on_colormap_changed
                )

                self.draw_line_between_components()

            except Exception as e:  # noqa: BLE001
                print(f"Error applying saved colormap settings: {e}")
                try:
                    self.comp1_fractions_layer.events.colormap.connect(
                        self._on_colormap_changed
                    )
                    self.comp1_fractions_layer.events.contrast_limits.connect(
                        self._on_contrast_limits_changed
                    )
                    self.comp1_fractions_layer.events.gamma.connect(
                        self._on_colormap_changed
                    )
                except Exception:  # noqa: BLE001
                    pass

    def get_all_artists(self):
        """Get all matplotlib artists."""
        artists = []
        for comp in self.components:
            if comp is not None:
                if comp.dot is not None:
                    artists.append(comp.dot)
                if comp.text is not None:
                    artists.append(comp.text)
        if self.component_line is not None:
            artists.append(self.component_line)
        if self.component_polygon is not None:
            artists.append(self.component_polygon)
        if self.component_histogram is not None:
            hist_artists = self.component_histogram
            if not isinstance(hist_artists, (list, tuple)):
                hist_artists = [hist_artists]
            artists.extend(hist_artists)
        return artists

    def set_artists_visible(self, visible):
        """Set visibility of all artists created by this widget."""
        for comp in self.components:
            if comp is not None:
                if comp.dot is not None:
                    comp.dot.set_visible(visible and self.show_component_dots)
                if comp.text is not None:
                    comp.text.set_visible(visible)
        if self.component_line is not None:
            self.component_line.set_visible(visible)
        if self.component_polygon is not None:
            self.component_polygon.set_visible(visible)
        if self.component_histogram is not None:
            hist_artists = self.component_histogram
            if not isinstance(hist_artists, (list, tuple)):
                hist_artists = [hist_artists]
            for artist in hist_artists:
                artist.set_visible(visible)

    def clear_artists(self):
        """Clear (remove) all artists created by this widget."""
        self._clear_components()

    def _toggle_plot_section(self, checked):
        """Toggle visibility of the plot section."""
        self.plot_section.setVisible(checked)

    def _on_plot_setting_changed(self):
        """Handle changes to plot settings from dialog."""
        if hasattr(self, 'colormap_line_checkbox'):
            self.show_colormap_line = self.colormap_line_checkbox.isChecked()
            self._update_components_setting_in_metadata(
                'two_component_line_settings.show_colormap_line',
                self.show_colormap_line,
            )

        if hasattr(self, 'show_dots_checkbox'):
            self.show_component_dots = self.show_dots_checkbox.isChecked()
            self._update_components_setting_in_metadata(
                'two_component_line_settings.show_component_dots',
                self.show_component_dots,
            )

        if hasattr(self, 'fraction_histogram_checkbox'):
            self.show_fraction_histogram = (
                self.fraction_histogram_checkbox.isChecked()
            )
            self._update_components_setting_in_metadata(
                'two_component_line_settings.show_fraction_histogram',
                self.show_fraction_histogram,
            )

        active_components = [
            c for c in self.components if c is not None and c.dot is not None
        ]

        if (
            len(active_components) == 2
            and self.show_colormap_line
            and self.fractions_colormap is not None
        ) or len(active_components) > 2:
            self._update_component_colors()
        else:
            for comp in self.components:
                if comp.dot is not None:
                    comp.dot.set_color(self.default_component_color)

        components_tab_is_active = (
            self.parent_widget is not None
            and getattr(self.parent_widget, "tab_widget", None) is not None
            and self.parent_widget.tab_widget.currentWidget()
            is self.parent_widget.components_tab
        )
        self.set_artists_visible(components_tab_is_active)

        self.draw_line_between_components()
        if self.parent_widget is not None:
            self.parent_widget.canvas_widget.canvas.draw_idle()

    def _on_line_offset_changed(self, value):
        """Handle changes to line offset (float, data units)."""
        self.line_offset = float(value)
        self._update_components_setting_in_metadata(
            'two_component_line_settings.line_offset', self.line_offset
        )
        self.draw_line_between_components()
        if self.parent_widget is not None:
            self.parent_widget.canvas_widget.canvas.draw_idle()

    def _on_histogram_height_changed(self, value):
        """Handle changes to the fraction histogram overlay height (float)."""
        self.histogram_overlay_height = float(value)
        self._update_components_setting_in_metadata(
            'two_component_line_settings.histogram_overlay_height',
            self.histogram_overlay_height,
        )
        if self.show_fraction_histogram:
            self.draw_line_between_components()
            if self.parent_widget is not None:
                self.parent_widget.canvas_widget.canvas.draw_idle()

    def _on_histogram_offset_changed(self, value):
        """Handle changes to the fraction histogram overlay offset (float)."""
        self.histogram_offset = float(value)
        self._update_components_setting_in_metadata(
            'two_component_line_settings.histogram_offset',
            self.histogram_offset,
        )
        if self.show_fraction_histogram:
            self.draw_line_between_components()
            if self.parent_widget is not None:
                self.parent_widget.canvas_widget.canvas.draw_idle()

    def _on_histogram_transparency_changed(self, value):
        """Handle changes to the histogram transparency (0=opaque, 1=clear)."""
        self.histogram_alpha = 1.0 - float(value)
        self._update_components_setting_in_metadata(
            'two_component_line_settings.histogram_alpha',
            self.histogram_alpha,
        )
        if self.show_fraction_histogram:
            self.draw_line_between_components()
            if self.parent_widget is not None:
                self.parent_widget.canvas_widget.canvas.draw_idle()

    def _on_line_width_changed(self, value):
        """Handle changes to line width (float)."""
        self.line_width = float(value)
        self._update_components_setting_in_metadata(
            'two_component_line_settings.line_width', self.line_width
        )

        if isinstance(self.component_line, LineCollection):
            with contextlib.suppress(Exception):
                self.component_line.set_linewidths([self.line_width])

            if self.parent_widget is not None:
                self.parent_widget.canvas_widget.canvas.draw_idle()
        else:
            self.draw_line_between_components()
            if self.parent_widget is not None:
                self.parent_widget.canvas_widget.canvas.draw_idle()

    def _on_line_transparency_changed(self, value):
        """Handle changes to line transparency (0=opaque, 1=clear)."""
        self.line_alpha = 1.0 - float(value)
        self._update_components_setting_in_metadata(
            'two_component_line_settings.line_alpha', self.line_alpha
        )

        if self.component_line is not None and hasattr(
            self.component_line, 'set_alpha'
        ):
            self.component_line.set_alpha(self.line_alpha)

        for comp in self.components:
            if comp.dot is not None:
                comp.dot.set_alpha(self.line_alpha)

        self.draw_line_between_components()
        if self.parent_widget is not None:
            self.parent_widget.canvas_widget.canvas.draw_idle()

    def _toggle_style_section(self, checked):
        """Toggle visibility of the style section."""
        self.style_section.setVisible(checked)

    def _pick_label_color(self):
        """Open color dialog to pick label color."""
        color = QColorDialog.getColor()
        if color.isValid():
            self.label_color = color.name()
            self._update_components_setting_in_metadata(
                'two_components_label_settings.color', self.label_color
            )
            self._apply_styles_to_labels()

    def _on_label_style_changed(self):
        """Handle changes to label style settings."""
        self.label_fontsize = self.fontsize_spin.value()
        self.label_bold = self.bold_checkbox.isChecked()
        self.label_italic = self.italic_checkbox.isChecked()
        self._update_components_setting_in_metadata(
            'two_components_label_settings.fontsize', self.label_fontsize
        )
        self._update_components_setting_in_metadata(
            'two_components_label_settings.bold', self.label_bold
        )
        self._update_components_setting_in_metadata(
            'two_components_label_settings.italic', self.label_italic
        )
        self._apply_styles_to_labels()

    def _apply_styles_to_labels(self):
        """Apply current style settings to all component labels."""
        weight = 'bold' if self.label_bold else 'normal'
        style = 'italic' if self.label_italic else 'normal'
        for comp in self.components:
            if comp.text is not None:
                comp.text.set_fontsize(self.label_fontsize)
                comp.text.set_fontweight(weight)
                comp.text.set_fontstyle(style)
                comp.text.set_color(self.label_color)
        if self.parent_widget is not None:
            self.parent_widget.canvas_widget.canvas.draw_idle()

    def _update_lifetime_inputs_visibility(self):
        """Show/hide lifetime inputs depending on frequency availability."""
        has_freq = False

        try:
            has_freq = (
                self.parent_widget is not None
                and hasattr(self.parent_widget, '_get_frequency_from_layer')
                and self.parent_widget._get_frequency_from_layer() is not None
            )
        except (AttributeError, TypeError):
            has_freq = False

        for _i, comp in enumerate(self.components):
            if comp is not None and (
                hasattr(comp, 'ui_elements')
                and 'lifetime_label' in comp.ui_elements
            ):
                comp.ui_elements['lifetime_label'].setVisible(has_freq)

                if comp.lifetime_edit is not None:
                    comp.lifetime_edit.setVisible(has_freq)

                    from qtpy.QtWidgets import QSizePolicy

                    if has_freq:
                        comp.lifetime_edit.setSizePolicy(
                            QSizePolicy.Expanding, QSizePolicy.Fixed
                        )
                        comp.ui_elements['lifetime_label'].setSizePolicy(
                            QSizePolicy.Fixed, QSizePolicy.Fixed
                        )
                    else:
                        comp.lifetime_edit.setSizePolicy(
                            QSizePolicy.Ignored, QSizePolicy.Ignored
                        )
                        comp.ui_elements['lifetime_label'].setSizePolicy(
                            QSizePolicy.Ignored, QSizePolicy.Ignored
                        )

        if has_freq:
            for i, comp in enumerate(self.components):
                if (
                    comp is not None
                    and comp.lifetime_edit is not None
                    and comp.lifetime_edit.text().strip()
                ):
                    self._update_component_from_lifetime(i)

    def _update_component_input_styling(self, idx: int):
        """Update the styling of component input fields based on their state."""
        if idx >= len(self.components) or self.components[idx] is None:
            return

        if not self._analysis_attempted:
            return

        comp = self.components[idx]

        has_g_value = comp.g_edit.text().strip() != ""
        has_s_value = comp.s_edit.text().strip() != ""

        if has_g_value and has_s_value:
            comp.g_edit.setStyleSheet("")
            comp.s_edit.setStyleSheet("")
        else:
            if self._should_highlight_missing_components():
                comp.g_edit.setStyleSheet(
                    "background-color: #330000; border: 2px solid #cc0000;"
                )
                comp.s_edit.setStyleSheet(
                    "background-color: #330000; border: 2px solid #cc0000;"
                )
            else:
                comp.g_edit.setStyleSheet("")
                comp.s_edit.setStyleSheet("")

    def _should_highlight_missing_components(self):
        """Check if we should highlight missing component locations."""
        num_total_components = len(
            [c for c in self.components if c is not None]
        )

        if num_total_components < 2:
            return False

        required_harmonics = self._get_required_harmonics(num_total_components)

        if required_harmonics <= 1:
            return False

        harmonics_with_components = self._get_harmonics_with_components()

        return len(harmonics_with_components) < required_harmonics

    def _update_all_component_styling(self):
        """Update styling for all component input fields."""
        for comp in self.components:
            if comp is not None:
                self._update_component_input_styling(comp.idx)

    def _compute_phasor_from_lifetime(self, lifetime_text, harmonic: int = 1):
        """Compute (G,S) from lifetime string; return tuple or (None,None)."""
        try:
            lifetime = float(lifetime_text)
        except (TypeError, ValueError):
            return None, None
        freq = self.parent_widget._get_frequency_from_layer()
        if freq is None:
            return None, None

        re, im = phasor_from_lifetime(freq * harmonic, lifetime)
        if np.ndim(re) > 0:
            re = float(np.array(re).ravel()[0])
        if np.ndim(im) > 0:
            im = float(np.array(im).ravel()[0])
        return re, im

    def _update_component_from_lifetime(self, idx: int):
        """Update component G/S coordinates based on lifetime input for all available harmonics."""
        comp = self.components[idx]
        txt = comp.lifetime_edit.text().strip()

        if not txt:
            return

        try:
            lifetime = float(txt)
        except ValueError:
            return

        freq = self.parent_widget._get_frequency_from_layer()
        if freq is None:
            return

        available_harmonics = self._get_available_harmonics()
        if not available_harmonics:
            return

        self._updating_from_lifetime = True

        for harmonic in available_harmonics:
            re, im = phasor_from_lifetime(freq * harmonic, lifetime)
            if np.ndim(re) > 0:
                re = float(np.array(re).ravel()[0])
            if np.ndim(im) > 0:
                im = float(np.array(im).ravel()[0])

            self._update_component_gs_coords(idx, harmonic, re, im)
            self._update_component_lifetime(idx, harmonic, lifetime)

            if harmonic == getattr(self.parent_widget, 'harmonic', 1):
                comp.g_edit.setText(f"{re:.3f}")
                comp.s_edit.setText(f"{im:.3f}")
                self._on_component_coords_changed(idx)

        self._updating_from_lifetime = False

    def _get_lifetime_from_coords(self, x, y, harmonic):
        """Calculate normal lifetime from coordinates x, y at a given harmonic."""
        if self.parent_widget is None:
            return None
        try:
            freq = self.parent_widget._get_frequency_from_layer()
            if freq is None:
                return None
            frequency = freq * harmonic
            lifetime = phasor_to_normal_lifetime(x, y, frequency=frequency)
            if np.ndim(lifetime) > 0:
                lifetime = float(np.array(lifetime).ravel()[0])
            if np.isfinite(lifetime):
                return lifetime
        except (ValueError, TypeError, AttributeError):
            pass
        return None

    def _populate_select_menu(self, idx, menu):
        """Dynamically populate the select button menu with available cursor/intersect options."""
        menu.clear()

        # 1. Option: Select on plot
        select_plot_action = menu.addAction("Select on plot")
        select_plot_action.triggered.connect(
            lambda: self._select_component(idx)
        )

        # 2. Option: Select from phasor center
        select_center_action = menu.addAction(
            "Select from layer(s) phasor center"
        )
        select_center_action.triggered.connect(
            lambda _, curr_idx=idx: self._select_from_phasor_center(curr_idx)
        )

        # 3. Option: Select from cursor center (submenu)
        cursor_menu = menu.addMenu("Select from cursor center")

        # Gather all available cursors from SelectionWidget
        cursors_added = False
        selection_tab = getattr(self.parent_widget, 'selection_tab', None)
        if selection_tab is not None:
            # A. Cursor-selection cursors (circular / elliptical / polar)
            cursor_widget = getattr(
                selection_tab, 'cursor_selection_widget', None
            )
            if cursor_widget and hasattr(cursor_widget, '_cursors'):
                type_labels = {
                    'circular': 'Circular',
                    'elliptic': 'Elliptical',
                    'polar': 'Polar',
                }
                for i, cursor in enumerate(cursor_widget._cursors):
                    harmonic = cursor.get('harmonic', 1)
                    color = cursor.get('color')
                    cursor_type = cursor.get('type', 'circular')
                    if cursor_type == 'polar':
                        phase_min = cursor.get('phase_min')
                        phase_max = cursor.get('phase_max')
                        mod_min = cursor.get('modulation_min')
                        mod_max = cursor.get('modulation_max')
                        if any(
                            v is None
                            for v in [phase_min, phase_max, mod_min, mod_max]
                        ):
                            continue
                        phase_center = (phase_min + phase_max) / 2.0
                        mod_center = (mod_min + mod_max) / 2.0
                        g = mod_center * np.cos(np.deg2rad(phase_center))
                        s = mod_center * np.sin(np.deg2rad(phase_center))
                    else:
                        g = cursor.get('g')
                        s = cursor.get('s')
                        if g is None or s is None:
                            continue
                    label = type_labels.get(cursor_type, 'Cursor')
                    name = f"{label} {i + 1} (H{harmonic})"
                    text = f"{name}: G={g:.3f}, S={s:.3f}"
                    action = QWidgetAction(cursor_menu)
                    action.setText(text)
                    action_widget = ColorActionWidget(
                        text, color, action, cursor_menu
                    )
                    action.setDefaultWidget(action_widget)
                    action.triggered.connect(
                        lambda _, g_val=g, s_val=s: self._set_component_coords_from_menu(
                            idx, g_val, s_val
                        )
                    )
                    cursor_menu.addAction(action)
                    cursors_added = True

            # D. GMM Clusters
            cluster_widget = getattr(
                selection_tab, 'automatic_clustering_widget', None
            )
            if cluster_widget and hasattr(cluster_widget, '_clusters'):
                for i, cluster in enumerate(cluster_widget._clusters):
                    g = cluster.get('g')
                    s = cluster.get('s')
                    harmonic = cluster.get('harmonic', 1)
                    color = cluster.get('color')
                    if g is not None and s is not None:
                        name = f"Cluster {i + 1} (H{harmonic})"
                        text = f"{name}: G={g:.3f}, S={s:.3f}"
                        action = QWidgetAction(cursor_menu)
                        action.setText(text)
                        action_widget = ColorActionWidget(
                            text, color, action, cursor_menu
                        )
                        action.setDefaultWidget(action_widget)
                        action.triggered.connect(
                            lambda _, g_val=g, s_val=s: self._set_component_coords_from_menu(
                                idx, g_val, s_val
                            )
                        )
                        cursor_menu.addAction(action)
                        cursors_added = True

        if not cursors_added:
            empty_action = cursor_menu.addAction("No active cursors")
            empty_action.setEnabled(False)

        # 4. Option: Auto intersect (For all components except the first one)
        if idx > 0:
            auto_action = menu.addAction("Auto intersect semicircle")
            auto_action.triggered.connect(
                lambda _, curr_idx=idx: self._auto_place_component_by_index(
                    curr_idx
                )
            )

    def _set_component_coords_from_menu(self, idx, g, s):
        """Set component coordinates from a selection option, including lifetime calculation."""
        comp = self.components[idx]
        if comp is None:
            return

        comp.g_edit.setText(f"{g:.3f}")
        comp.s_edit.setText(f"{s:.3f}")

        current_harmonic = getattr(self.parent_widget, 'harmonic', 1)
        lifetime = self._get_lifetime_from_coords(g, s, current_harmonic)

        self._updating_from_lifetime = True
        try:
            if comp.lifetime_edit is not None:
                if lifetime is not None:
                    comp.lifetime_edit.setText(f"{lifetime:.3f}")
                else:
                    comp.lifetime_edit.clear()

            self._on_component_coords_changed(idx)
            self._update_component_lifetime(idx, current_harmonic, lifetime)
        finally:
            self._updating_from_lifetime = False

        self._redraw(force=True)

    def _on_component_coords_changed(self, idx: int):
        """Handle changes to component G/S coordinates from text inputs."""
        comp = self.components[idx]
        try:
            x = float(comp.g_edit.text())
            y = float(comp.s_edit.text())
        except ValueError:
            return

        current_harmonic = getattr(self.parent_widget, 'harmonic', 1)
        self._update_component_gs_coords(idx, current_harmonic, x, y)

        if comp.lifetime_edit is not None and not self._updating_from_lifetime:
            comp.lifetime_edit.clear()
            self._update_component_lifetime(idx, current_harmonic, None)

        if comp.dot is not None:
            comp.dot.set_data([x], [y])
            if comp.text is not None:
                ox, oy = comp.text_offset
                comp.text.set_position((x + ox, y + oy))
            self.draw_line_between_components()
        else:
            self._create_component_at_coordinates(idx, x, y)

    def _on_component_name_changed(self, idx: int):
        """Handle changes to component name."""
        comp = self.components[idx]
        name = comp.name_edit.text().strip()

        old_name = None
        if not self._updating_settings and self.current_image_layer_name:
            # Check if layer still exists (defensive check for cleanup/teardown)
            if self.current_image_layer_name not in self.viewer.layers:
                return
            layer = self.viewer.layers[self.current_image_layer_name]
            if (
                'settings' in layer.metadata
                and 'component_analysis' in layer.metadata['settings']
            ):

                idx_str = str(idx)
                if idx_str in layer.metadata['settings'][
                    'component_analysis'
                ].get('components', {}):
                    old_name = layer.metadata['settings'][
                        'component_analysis'
                    ]['components'][idx_str].get('name')

            if old_name != name:
                self._update_fraction_layer_names(idx, old_name, name)

            self._update_component_name(idx, name)

        if comp.dot is None:
            return

        prev_pos = None
        if comp.text is not None:
            prev_pos = comp.text.get_position()
            comp.text.remove()
            comp.text = None

        if name:
            dx, dy = comp.dot.get_data()
            if prev_pos is None:
                ox, oy = comp.text_offset
                base_x, base_y = dx[0] + ox, dy[0] + oy
            else:
                base_x, base_y = prev_pos
                comp.text_offset = (base_x - dx[0], base_y - dy[0])
            ax = self.parent_widget.canvas_widget.figure.gca()
            comp.text = ax.text(
                base_x,
                base_y,
                name,
                fontsize=self.label_fontsize,
                fontweight='bold' if self.label_bold else 'normal',
                fontstyle='italic' if self.label_italic else 'normal',
                color=self.label_color,
                picker=True,
            )

        self.parent_widget.canvas_widget.canvas.draw_idle()

    def _update_fraction_layer_names(
        self, idx: int, old_name: str, new_name: str
    ):
        """Update the names of the fraction layers when component names change."""
        if not self.current_image_layer_name:
            return

        # Only handle comp1 layer (idx == 0) for two-component linear projection
        # Component fit creates individual fraction layers per component
        if idx > 0 and self.analysis_type == "Linear Projection":
            return

        old_display_name = old_name if old_name else f"Component {idx + 1}"
        new_display_name = new_name if new_name else f"Component {idx + 1}"

        old_layer_name = (
            f"{old_display_name} fractions: {self.current_image_layer_name}"
        )
        new_layer_name = (
            f"{new_display_name} fractions: {self.current_image_layer_name}"
        )

        if (
            old_layer_name in self.viewer.layers
            and old_layer_name != new_layer_name
        ):
            layer_obj = self.viewer.layers[old_layer_name]
            layer_obj.name = new_layer_name

            if idx == 0:
                self.comp1_fractions_layer = layer_obj

        elif new_layer_name not in self.viewer.layers:
            possible_old_names = [
                f"Component {idx + 1} fractions: {self.current_image_layer_name}",
                (
                    f"{old_display_name} fractions: {self.current_image_layer_name}"
                    if old_name
                    else None
                ),
            ]

            for possible_old_name in possible_old_names:
                if (
                    possible_old_name
                    and possible_old_name in self.viewer.layers
                ):
                    layer_obj = self.viewer.layers[possible_old_name]
                    layer_obj.name = new_layer_name

                    if idx == 0:
                        self.comp1_fractions_layer = layer_obj
                    break

    def _create_component_at_coordinates(self, idx: int, x: float, y: float):
        """Create a component dot and label at specified coordinates."""
        if self.parent_widget is None:
            return
        comp = self.components[idx]

        active_components = [
            c for c in self.components if c is not None and c.dot is not None
        ]

        active_count_after = len(active_components) + 1

        try:
            colors = self._get_component_colors_for_count(active_count_after)

            if len(colors) > idx:
                color = colors[idx]
            else:
                color = self.component_colors[idx % len(self.component_colors)]

        except Exception:  # noqa: BLE001
            color = self.component_colors[idx % len(self.component_colors)]

        ax = self.parent_widget.canvas_widget.figure.gca()
        comp.dot = ax.plot(
            x,
            y,
            'o',
            color=color,
            markersize=8,
            label=comp.label,
            alpha=self.line_alpha,
            markeredgewidth=0,
            zorder=11,
        )[0]
        name = comp.name_edit.text().strip()
        if name:
            ox, oy = comp.text_offset
            comp.text = ax.text(
                x + ox,
                y + oy,
                name,
                fontsize=self.label_fontsize,
                fontweight='bold' if self.label_bold else 'normal',
                fontstyle='italic' if self.label_italic else 'normal',
                color=self.label_color,
                picker=True,
                zorder=12,
            )
        self._make_components_draggable()
        self.parent_widget.canvas_widget.canvas.draw_idle()
        self.draw_line_between_components()

    def _get_component_coords_for_harmonic(self, harmonic):
        """Get component coordinates for a specific harmonic from metadata."""
        component_g = []
        component_s = []
        component_names = []

        current_harmonic = getattr(self.parent_widget, 'harmonic', 1)

        if harmonic == current_harmonic:
            for comp in self.components:
                if comp is not None and comp.dot is not None:
                    x_data, y_data = comp.dot.get_data()
                    component_g.append(x_data[0])
                    component_s.append(y_data[0])
                    name = comp.name_edit.text().strip()
                    if not name:
                        name = comp.label
                    component_names.append(name)
        else:
            if not self.current_image_layer_name:
                return component_g, component_s, component_names

            # Check if layer still exists (defensive check for cleanup/teardown)
            if self.current_image_layer_name not in self.viewer.layers:
                return component_g, component_s, component_names

            layer = self.viewer.layers[self.current_image_layer_name]
            if (
                'settings' not in layer.metadata
                or 'component_analysis' not in layer.metadata['settings']
            ):
                return component_g, component_s, component_names

            settings = layer.metadata['settings']['component_analysis']
            components_data = settings.get('components', {})
            harmonic_key = str(harmonic)

            sorted_indices = sorted([int(k) for k in components_data])

            for idx in sorted_indices:
                idx_str = str(idx)
                comp_data = components_data[idx_str]
                gs_harmonics = comp_data.get('gs_harmonics', {})

                if harmonic_key in gs_harmonics:
                    harmonic_data = gs_harmonics[harmonic_key]
                    g = harmonic_data.get('g')
                    s = harmonic_data.get('s')

                    if g is not None and s is not None:
                        component_g.append(g)
                        component_s.append(s)

                        name = comp_data.get('name', '')
                        if not name:
                            name = f"Component {idx + 1}"
                        component_names.append(name)

        return component_g, component_s, component_names

    def _update_component_visibility(self):
        """Update visibility of component controls."""
        for _, comp in enumerate(self.components):
            if comp is not None and hasattr(comp, 'ui_elements'):
                comp_layout = comp.ui_elements['comp_layout']
                for j in range(comp_layout.count()):
                    item = comp_layout.itemAt(j)
                    if item.widget():
                        item.widget().setVisible(True)

    def _select_component(self, idx):
        """Prepare for selecting a component by clicking on the plot."""
        if self.parent_widget is None:
            return

        # If there's an active select listener, disconnect it first
        self._cancel_active_selection()

        comp = self.components[idx]
        if comp is None:
            return

        if comp.dot is not None:
            comp.dot.set_visible(False)
        if comp.text is not None:
            comp.text.set_visible(False)
        if self.component_line is not None:
            self.component_line.set_visible(False)
        self._redraw(force=True)

        original_text = "Select"
        comp.select_button.setText("Click on plot...")
        comp.select_button.setEnabled(False)

        # Store original text and idx so we can restore them later
        self._active_select_original_text = original_text
        self._active_select_idx = idx

        # Connect click event
        self._active_select_cid = (
            self.parent_widget.canvas_widget.canvas.mpl_connect(
                'button_press_event', self._handle_component_selection_event
            )
        )

        # Connect key press event (Esc key)
        self._active_select_key_cid = (
            self.parent_widget.canvas_widget.canvas.mpl_connect(
                'key_press_event', self._handle_select_key_press_event
            )
        )

        # Connect Escape key shortcut at Qt level so it cancels even if canvas does not have focus
        self._active_select_shortcut = QShortcut(
            QKeySequence(Qt.Key_Escape), self
        )
        self._active_select_shortcut.activated.connect(
            self._cancel_active_selection
        )

        # Focus the canvas so key_press_event has a chance of working as well
        with contextlib.suppress(Exception):
            self.parent_widget.canvas_widget.canvas.setFocus()

    def _handle_select_key_press_event(self, event):
        """Handle key press events during component selection on plot."""
        if event.key == 'escape':
            self._cancel_active_selection()

    def _cancel_active_selection(self):
        """Cancel the active plot selection and restore button states."""
        if self.parent_widget is None:
            return

        if self._active_select_cid is not None:
            with contextlib.suppress(Exception):
                self.parent_widget.canvas_widget.canvas.mpl_disconnect(
                    self._active_select_cid
                )
            self._active_select_cid = None

        if self._active_select_key_cid is not None:
            with contextlib.suppress(Exception):
                self.parent_widget.canvas_widget.canvas.mpl_disconnect(
                    self._active_select_key_cid
                )
            self._active_select_key_cid = None

        if getattr(self, '_active_select_shortcut', None) is not None:
            self._active_select_shortcut.setEnabled(False)
            self._active_select_shortcut.setParent(None)
            self._active_select_shortcut = None

        if getattr(self, '_active_select_idx', None) is not None:
            idx = self._active_select_idx
            comp = self.components[idx]
            if comp is not None:
                comp.select_button.setText(
                    getattr(self, '_active_select_original_text', "Select")
                )
                comp.select_button.setEnabled(True)

                if comp.dot is not None:
                    comp.dot.set_visible(self.show_component_dots)
                    if comp.text is not None:
                        comp.text.set_visible(True)
                self.draw_line_between_components()
                self._redraw(force=True)

            self._active_select_idx = None

    def _handle_component_selection_event(self, event):
        """Wrapper event handler for clicking on the plot to select a component."""
        if not event.inaxes:
            return

        idx = getattr(self, '_active_select_idx', None)
        if idx is None:
            return

        x, y = event.xdata, event.ydata
        comp = self.components[idx]
        if comp is None:
            return

        # Disable listeners first
        cid = self._active_select_cid
        key_cid = self._active_select_key_cid
        shortcut = getattr(self, '_active_select_shortcut', None)
        original_text = getattr(self, '_active_select_original_text', "Select")

        self._active_select_cid = None
        self._active_select_key_cid = None
        self._active_select_shortcut = None
        self._active_select_idx = None

        if cid is not None:
            with contextlib.suppress(Exception):
                self.parent_widget.canvas_widget.canvas.mpl_disconnect(cid)
        if key_cid is not None:
            with contextlib.suppress(Exception):
                self.parent_widget.canvas_widget.canvas.mpl_disconnect(key_cid)
        if shortcut is not None:
            shortcut.setEnabled(False)
            shortcut.setParent(None)

        # Set values
        comp.g_edit.setText(f"{x:.3f}")
        comp.s_edit.setText(f"{y:.3f}")

        current_harmonic = getattr(self.parent_widget, 'harmonic', 1)
        lifetime = self._get_lifetime_from_coords(x, y, current_harmonic)

        if comp.lifetime_edit is not None:
            if lifetime is not None:
                comp.lifetime_edit.setText(f"{lifetime:.3f}")
            else:
                comp.lifetime_edit.clear()

        self._update_component_gs_coords(idx, current_harmonic, x, y)
        self._update_component_lifetime(idx, current_harmonic, lifetime)

        was_new_location = comp.dot is None

        if comp.dot is None:
            self._create_component_at_coordinates(idx, x, y)
        else:
            comp.dot.set_data([x], [y])
            comp.dot.set_visible(self.show_component_dots)
            comp.dot.set_markeredgewidth(0)
            name = comp.name_edit.text().strip()
            if name:
                if comp.text is None:
                    ax = self.parent_widget.canvas_widget.figure.gca()
                    ox, oy = comp.text_offset
                    comp.text = ax.text(
                        x + ox,
                        y + oy,
                        name,
                        fontsize=self.label_fontsize,
                        fontweight='bold' if self.label_bold else 'normal',
                        fontstyle=(
                            'italic' if self.label_italic else 'normal'
                        ),
                        color=self.label_color,
                        picker=True,
                    )
                else:
                    ox, oy = comp.text_offset
                    comp.text.set_position((x + ox, y + oy))
                    comp.text.set_visible(True)

            self.draw_line_between_components()

        comp.select_button.setText(original_text)
        comp.select_button.setEnabled(True)

        if self._analysis_attempted:
            self._update_all_component_styling()

        self._redraw(force=True)

        if was_new_location:
            self._activate_next_unselected_component()

    def _activate_next_unselected_component(self):
        """Automatically activate the next component that doesn't have a location set."""
        visible_components = [c for c in self.components if c is not None]

        for comp in visible_components:
            if comp.dot is None:
                self._select_component(comp.idx)
                return

    def _get_default_colormap_max_colors(self, num_components):
        """Get the maximum value colors from the colormaps for components."""
        colors = []
        for i in range(num_components):
            colormap_name = self.component_colormap_names[
                i % len(self.component_colormap_names)
            ]

            try:
                cmap = plt.get_cmap(colormap_name)
                max_color_rgba = cmap(1.0)
                max_color_hex = mcolors.to_hex(max_color_rgba)
                colors.append(max_color_hex)
            except Exception:  # noqa: BLE001
                colors.append(
                    self.component_colors[i % len(self.component_colors)]
                )

        return colors

    def _get_component_colors(self):
        """Get colors for components based on the colormap ends or default colors."""
        active_components = [
            c for c in self.components if c is not None and c.dot is not None
        ]
        return self._get_component_colors_for_count(len(active_components))

    def _get_component_colors_for_count(self, num_components):
        """Get colors for a specific number of components (used for out-of-order selection)."""
        if not self.show_colormap_line:
            return [self.default_component_color] * max(
                num_components, len(self.components)
            )

        if num_components == 2:
            # Only use fractions colormap for Linear Projection mode
            if (
                self.analysis_type == "Linear Projection"
                and hasattr(self, 'fractions_colormap')
                and self.fractions_colormap is not None
            ):
                if (
                    hasattr(self, 'colormap_contrast_limits')
                    and self.colormap_contrast_limits is not None
                ):
                    vmin, vmax = self.colormap_contrast_limits
                else:
                    vmin, vmax = 0, 1

                if vmax > vmin:
                    component1_idx = int(
                        ((1.0 - vmin) / (vmax - vmin))
                        * (len(self.fractions_colormap) - 1)
                    )
                    component2_idx = int(
                        ((0.0 - vmin) / (vmax - vmin))
                        * (len(self.fractions_colormap) - 1)
                    )

                    component1_idx = max(
                        0,
                        min(len(self.fractions_colormap) - 1, component1_idx),
                    )
                    component2_idx = max(
                        0,
                        min(len(self.fractions_colormap) - 1, component2_idx),
                    )

                    component1_color = self.fractions_colormap[component1_idx]
                    component2_color = self.fractions_colormap[component2_idx]
                else:
                    component1_color = self.fractions_colormap[-1]
                    component2_color = self.fractions_colormap[0]

                return [component1_color, component2_color]
            else:
                # For Component Fit mode, extract colors from actual fraction layers if they exist
                if len(self.fraction_layers) >= 2:
                    colors = []
                    for i in range(2):
                        if i < len(self.fraction_layers):
                            layer = self.fraction_layers[i]
                            try:
                                cmap = layer.colormap
                                colormap_name = str(cmap)

                                if 'green' in colormap_name.lower():
                                    colors.append('green')
                                else:
                                    if callable(cmap):
                                        max_color = cmap(1.0)
                                    elif (
                                        hasattr(cmap, 'colors')
                                        and cmap.colors is not None
                                    ):
                                        max_color = cmap.colors[-1]
                                    else:
                                        mpl_cmap = plt.get_cmap(str(cmap))
                                        max_color = mpl_cmap(1.0)
                                    colors.append(max_color)
                            except Exception:  # noqa: BLE001
                                default_colors = (
                                    self._get_default_colormap_max_colors(2)
                                )
                                if i < len(default_colors):
                                    colors.append(default_colors[i])
                                else:
                                    colors.append(
                                        self.component_colors[
                                            i % len(self.component_colors)
                                        ]
                                    )
                        else:
                            default_colors = (
                                self._get_default_colormap_max_colors(2)
                            )
                            if i < len(default_colors):
                                colors.append(default_colors[i])
                            else:
                                colors.append(
                                    self.component_colors[
                                        i % len(self.component_colors)
                                    ]
                                )
                    return colors
                else:
                    return self._get_default_colormap_max_colors(2)

        elif num_components > 2 and len(self.fraction_layers) > 0:
            colors = []

            max_idx = max(len(self.components), num_components)

            for i in range(max_idx):
                if i < len(self.fraction_layers):
                    layer = self.fraction_layers[i]
                    try:
                        cmap = layer.colormap
                        colormap_name = str(cmap)

                        if 'green' in colormap_name.lower():
                            colors.append('green')
                        else:
                            if callable(cmap):

                                max_color = cmap(1.0)
                            elif (
                                hasattr(cmap, 'colors')
                                and cmap.colors is not None
                            ):
                                max_color = cmap.colors[-1]
                            else:
                                mpl_cmap = plt.get_cmap(str(cmap))
                                max_color = mpl_cmap(1.0)
                            colors.append(max_color)
                    except Exception:  # noqa: BLE001
                        default_colors = self._get_default_colormap_max_colors(
                            max_idx
                        )
                        if i < len(default_colors):
                            colors.append(default_colors[i])
                        else:
                            colors.append(
                                self.component_colors[
                                    i % len(self.component_colors)
                                ]
                            )
                else:
                    default_colors = self._get_default_colormap_max_colors(
                        max_idx
                    )
                    if i < len(default_colors):
                        colors.append(default_colors[i])
                    else:
                        colors.append(
                            self.component_colors[
                                i % len(self.component_colors)
                            ]
                        )
            return colors
        else:
            max_idx = max(len(self.components), num_components, 1)
            return self._get_default_colormap_max_colors(max_idx)

    def _update_component_colors(self):
        """Update the colors of the component dots to match colormap ends."""
        active_components = [
            c for c in self.components if c is not None and c.dot is not None
        ]

        if len(active_components) == 0:
            return

        colors = self._get_component_colors_for_count(len(active_components))

        if len(active_components) == 2:
            comp_indices = [c.idx for c in active_components]
            comp_indices.sort()

            for i, comp_idx in enumerate(comp_indices):
                if (
                    comp_idx < len(self.components)
                    and self.components[comp_idx].dot is not None
                    and i < len(colors)
                ):
                    self.components[comp_idx].dot.set_color(colors[i])
        else:
            for comp in self.components:
                if (
                    comp is not None
                    and comp.dot is not None
                    and comp.idx < len(colors)
                ):
                    comp.dot.set_color(colors[comp.idx])

        if self.parent_widget is not None:
            self.parent_widget.canvas_widget.canvas.draw_idle()

    def draw_line_between_components(self):
        """Draw a line between for two components, or polygon for 3+ components."""
        active_components = [
            c for c in self.components if c is not None and c.dot is not None
        ]

        if len(active_components) < 2:
            return

        if self.component_line is not None:
            with contextlib.suppress(ValueError, AttributeError):
                self.component_line.remove()
            self.component_line = None

        if self.component_polygon is not None:
            with contextlib.suppress(ValueError, AttributeError):
                self.component_polygon.remove()
            self.component_polygon = None

        self._remove_histogram_overlay()

        try:
            if len(active_components) >= 3:
                self._update_polygon()
                components_tab_is_active = (
                    self.parent_widget is not None
                    and getattr(self.parent_widget, "tab_widget", None)
                    is not None
                    and self.parent_widget.tab_widget.currentWidget()
                    is self.parent_widget.components_tab
                )
                self.set_artists_visible(components_tab_is_active)
                return

            if not all(c.dot is not None for c in self.components[:2]):
                return

            x1_data, y1_data = self.components[0].dot.get_data()
            x2_data, y2_data = self.components[1].dot.get_data()
            ox1, oy1 = x1_data[0], y1_data[0]
            ox2, oy2 = x2_data[0], y2_data[0]

            if self.line_offset != 0.0:
                vx = ox2 - ox1
                vy = oy2 - oy1
                length = np.hypot(vx, vy)
                if length > 0:
                    nx = -vy / length
                    ny = vx / length
                    ox1 += nx * self.line_offset
                    oy1 += ny * self.line_offset
                    ox2 += nx * self.line_offset
                    oy2 += ny * self.line_offset

            ax = self.parent_widget.canvas_widget.figure.gca()

            use_colormap = (
                self.analysis_type == "Linear Projection"
                and self.show_colormap_line
                and self.comp1_fractions_layer is not None
                and self.fractions_colormap is not None
            )

            if use_colormap:
                self._draw_colormap_line(ax, ox1, oy1, ox2, oy2)
                self._update_component_colors()
                if hasattr(self.component_line, "set_capstyle"):
                    with contextlib.suppress(Exception):
                        self.component_line.set_capstyle('butt')

                if hasattr(self.component_line, 'set_zorder'):
                    with contextlib.suppress(Exception):
                        self.component_line.set_zorder(10)
            else:
                self.component_line = ax.plot(
                    [ox1, ox2],
                    [oy1, oy2],
                    color=self.default_component_color,
                    linewidth=self.line_width,
                    alpha=self.line_alpha,
                    zorder=10,
                )[0]
                if hasattr(self.component_line, "set_solid_capstyle"):
                    with contextlib.suppress(Exception):
                        self.component_line.set_solid_capstyle('butt')

                self._update_component_colors()

            if (
                self.show_fraction_histogram
                and self.analysis_type == "Linear Projection"
                and self.comp1_fractions_layer is not None
                and self.fractions_colormap is not None
            ):
                self._draw_fraction_histogram_overlay(ax, ox1, oy1, ox2, oy2)

            self.parent_widget.canvas_widget.canvas.draw_idle()

            components_tab_is_active = (
                self.parent_widget is not None
                and getattr(self.parent_widget, "tab_widget", None) is not None
                and self.parent_widget.tab_widget.currentWidget()
                is self.parent_widget.components_tab
            )
            self.set_artists_visible(components_tab_is_active)

        except Exception as e:  # noqa: BLE001
            show_error(f"Error drawing line/polygon: {str(e)}")

    def _draw_colormap_line(self, ax, x1, y1, x2, y2):
        """Draw a colormap bar between two components."""
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        if length == 0:
            return

        t_values = np.linspace(0, 1, 500)
        trajectory_real = x1 + t_values * dx
        trajectory_imag = y1 + t_values * dy

        density_factor = 2
        num_segments = min(
            len(trajectory_real) * density_factor, len(trajectory_real) - 1
        )

        if self.fractions_colormap is not None:
            if len(self.fractions_colormap) <= 32:
                colormap = LinearSegmentedColormap.from_list(
                    "fractions_interp", self.fractions_colormap, N=256
                )
            else:
                colormap = ListedColormap(self.fractions_colormap)
        else:
            colormap = plt.cm.jet

        if (
            hasattr(self, 'colormap_contrast_limits')
            and self.colormap_contrast_limits is not None
        ):
            vmin, vmax = self.colormap_contrast_limits
        elif self.comp1_fractions_layer is not None:
            vmin, vmax = self.comp1_fractions_layer.contrast_limits
        else:
            vmin, vmax = 0, 1

        segments = []
        colors = []

        for i in range(num_segments):
            start_idx = int(i * (len(trajectory_real) - 1) / num_segments)
            end_idx = int((i + 1) * (len(trajectory_real) - 1) / num_segments)

            end_idx = min(end_idx, len(trajectory_real) - 1)

            if i > 0:
                start_idx = max(0, start_idx - 1)

            segment = [
                (trajectory_real[start_idx], trajectory_imag[start_idx]),
                (trajectory_real[end_idx], trajectory_imag[end_idx]),
            ]
            segments.append(segment)

            t = (
                start_idx / (len(trajectory_real) - 1)
                if len(trajectory_real) > 1
                else 0
            )
            fraction_value = 1.0 - t
            colors.append(fraction_value)

        lc = LineCollection(
            segments, cmap=colormap, linewidths=self.line_width
        )
        lc.set_array(np.array(colors))
        gamma = getattr(self, 'fractions_gamma', 1.0) or 1.0
        if gamma != 1.0 and vmax > vmin:
            # Match the fraction layer's gamma so the phasor-plot gradient
            # reads the same as the image and the histogram.
            lc.set_norm(PowerNorm(gamma, vmin=vmin, vmax=vmax))
        else:
            lc.set_clim(vmin, vmax)
        lc.set_alpha(self.line_alpha)

        if hasattr(lc, "set_capstyle"):
            with contextlib.suppress(Exception):
                lc.set_capstyle('butt')
        self.component_line = ax.add_collection(lc)

    def _remove_histogram_overlay(self):
        """Remove the fraction histogram overlay artists, if present."""
        if self.component_histogram is None:
            return
        artists = self.component_histogram
        if not isinstance(artists, (list, tuple)):
            artists = [artists]
        for artist in artists:
            with contextlib.suppress(ValueError, AttributeError):
                artist.remove()
        self.component_histogram = None

    def _get_first_component_fraction_values(self):
        """Return the pooled first-component fraction values for the histogram.

        Pools the current (possibly range-clipped) fraction data across every
        image layer analyzed for the first component so the overlay tracks the
        histogram widget's range slider, matching the merged fraction histogram.
        """
        comp1_name = None
        if self.components and self.components[0] is not None:
            comp1_name = self.components[0].name_edit.text().strip()
        if not comp1_name:
            comp1_name = "Component 1"

        layers = self._get_fraction_layers_for_component(comp1_name)
        arrays = []
        for fl in layers.values():
            arrays.append(np.asarray(fl.data, dtype=float).ravel())

        if not arrays and self.comp1_fractions_layer is not None:
            arrays.append(
                np.asarray(
                    self.comp1_fractions_layer.data, dtype=float
                ).ravel()
            )

        if not arrays:
            return np.empty(0)

        pooled = np.concatenate(arrays)
        return pooled[np.isfinite(pooled)]

    def _draw_fraction_histogram_overlay(self, ax, x1, y1, x2, y2):
        """Overlay the first component's fraction histogram on the line.

        Delegates to the shared, stateless
        :func:`draw_fraction_histogram_overlay` so the interactive plot and the
        batch-analysis export render identically.
        """
        if self.comp1_fractions_layer is None:
            return

        values = self._get_first_component_fraction_values()
        artists = draw_fraction_histogram_overlay(
            ax,
            x1,
            y1,
            x2,
            y2,
            values,
            self.fractions_colormap,
            height=self.histogram_overlay_height,
            offset=self.histogram_offset,
            alpha=self.histogram_alpha,
            contrast_limits=self.colormap_contrast_limits,
            gamma=getattr(self, 'fractions_gamma', 1.0),
        )
        if artists:
            self.component_histogram = artists

    def _update_polygon(self):
        """Update polygon for multi-component visualization."""
        if self.component_polygon is not None:
            with contextlib.suppress(ValueError, AttributeError):
                self.component_polygon.remove()
            self.component_polygon = None

        active_components = [
            c for c in self.components if c is not None and c.dot is not None
        ]
        if len(active_components) < 3:
            return

        if self.parent_widget is None:
            return

        ax = self.parent_widget.canvas_widget.figure.gca()

        coords = []
        for comp in active_components:
            x_data, y_data = comp.dot.get_data()
            coords.append([x_data[0], y_data[0]])

        coords = np.array(coords)

        polygon = plt.Polygon(
            coords,
            fill=False,
            edgecolor=self.default_component_color,
            linewidth=self.line_width,
            alpha=self.line_alpha,
            zorder=10,
        )
        self.component_polygon = ax.add_patch(polygon)
        self.parent_widget.canvas_widget.canvas.draw_idle()

    def _on_colormap_changed(self, event):
        """Handle changes to colormap of any fraction layer (linear projection or component fit)."""
        layer = event.source

        # Prevent recursion when we're updating linked layers
        if getattr(self, '_updating_linked_layers', False):
            return

        comp_idx = self._find_component_index_for_layer(layer)
        if comp_idx is None:
            return

        # The phasor-plot line/overlay gradient tracks the FIRST component's
        # colormap. With multiple analyzed images there is one fraction layer
        # per image, so update the gradient for ANY first-component layer -- not
        # only the specific one stored in ``comp1_fractions_layer`` (which would
        # otherwise leave the line stale when a different image's layer changes).
        if comp_idx == 0:
            self.fractions_colormap = layer.colormap.colors
            self.colormap_contrast_limits = layer.contrast_limits
            self.fractions_gamma = layer.gamma

        colormap_name = getattr(layer.colormap, 'name', 'custom')
        is_standard_colormap = self._is_standard_colormap(colormap_name)

        if is_standard_colormap:
            colormap_colors = None
        else:
            colormap_colors = layer.colormap.colors
            if colormap_colors is not None and (
                hasattr(colormap_colors, 'tolist')
                or isinstance(colormap_colors, np.ndarray)
            ):
                colormap_colors = colormap_colors.tolist()

        current_harmonic = getattr(self.parent_widget, 'harmonic', 1)
        self._update_component_colormap(
            comp_idx,
            current_harmonic,
            colormap_name if is_standard_colormap else None,
            colormap_colors,
            tuple(layer.contrast_limits),
        )

        # Sync colormap / gamma to all other layers belonging to the same
        # component so the derived layers of every analyzed image stay in step.
        self._sync_component_layers_colormap(comp_idx, layer.colormap)
        self._sync_component_layers_gamma(comp_idx, layer.gamma)

        self._update_component_colors()
        self.draw_line_between_components()

        # Refresh histogram if the changed layer is the currently displayed one
        selected_component = ""
        if hasattr(self, 'histogram_component_combobox'):
            selected_component = (
                self.histogram_component_combobox.currentText().strip()
            )

        comp = self.components[comp_idx]
        changed_component = (
            comp.name_edit.text().strip() or f"Component {comp_idx + 1}"
        )

        if (
            selected_component == changed_component
            and hasattr(self, 'histogram_widget')
            and self.histogram_widget is not None
        ):
            self.histogram_widget.update_colormap(
                colormap_colors=layer.colormap.colors,
                contrast_limits=list(layer.contrast_limits),
                gamma=layer.gamma,
            )
        else:
            self._refresh_second_component_colormap(
                comp_idx, selected_component, layer
            )

    def _on_contrast_limits_changed(self, event):
        """Handle changes to contrast limits of any fraction layer."""
        layer = event.source

        # Prevent recursion when we're updating linked layers
        if getattr(self, '_updating_linked_layers', False):
            return

        comp_idx = self._find_component_index_for_layer(layer)
        if comp_idx is None:
            return

        # Keep the line gradient's contrast in sync for any first-component
        # fraction layer, not just the one stored in ``comp1_fractions_layer``.
        if comp_idx == 0:
            self.colormap_contrast_limits = layer.contrast_limits

        contrast_limits = layer.contrast_limits
        if hasattr(contrast_limits, 'tolist') or isinstance(
            contrast_limits, np.ndarray
        ):
            contrast_limits = contrast_limits.tolist()
        else:
            contrast_limits = list(contrast_limits)

        current_harmonic = getattr(self.parent_widget, 'harmonic', 1)

        if not self._updating_settings and self.current_image_layer_name:
            # Check if layer still exists (defensive check for cleanup/teardown)
            if self.current_image_layer_name not in self.viewer.layers:
                return

            layer_obj = self.viewer.layers[self.current_image_layer_name]
            settings = layer_obj.metadata.get('settings', {}).get(
                'component_analysis', {}
            )
            idx_str = str(comp_idx)
            harmonic_key = str(current_harmonic)

            if idx_str in settings.get(
                'components', {}
            ) and harmonic_key in settings['components'][idx_str].get(
                'gs_harmonics', {}
            ):

                harmonic_data = settings['components'][idx_str][
                    'gs_harmonics'
                ][harmonic_key]
                existing_colormap_name = harmonic_data.get('colormap_name')
                existing_colormap_colors = harmonic_data.get('colormap_colors')

                self._update_component_colormap(
                    comp_idx,
                    current_harmonic,
                    existing_colormap_name,
                    existing_colormap_colors,
                    tuple(contrast_limits),
                )

        self._sync_component_layers_contrast_limits(
            comp_idx, layer.contrast_limits
        )

        self.draw_line_between_components()

        # Refresh histogram only when the changed layer belongs to the
        # currently selected histogram component.
        selected_component = ""
        if hasattr(self, 'histogram_component_combobox'):
            selected_component = (
                self.histogram_component_combobox.currentText().strip()
            )

        comp = self.components[comp_idx]
        changed_component = (
            comp.name_edit.text().strip() or f"Component {comp_idx + 1}"
        )

        if (
            selected_component == changed_component
            and hasattr(self, 'histogram_widget')
            and self.histogram_widget is not None
        ):
            self.histogram_widget.update_colormap(
                colormap_colors=layer.colormap.colors,
                contrast_limits=contrast_limits,
                gamma=layer.gamma,
            )
        else:
            self._refresh_second_component_colormap(
                comp_idx, selected_component, layer
            )

    def _refresh_second_component_colormap(
        self, comp_idx, selected_component, layer
    ):
        """Refresh the histogram colormap when the second component is shown.

        In Linear Projection the second component is displayed using the first
        component's layer with an inverted fraction. When the first component's
        colormap or contrast limits change, mirror that change (reversed) onto
        the histogram if the second component is the active selection.
        """
        if (
            not hasattr(self, 'histogram_widget')
            or self.histogram_widget is None
        ):
            return

        name1, name2 = self._linear_projection_component_names()
        if name1 is None or comp_idx != 0 or selected_component != name2:
            return

        limits = list(layer.contrast_limits)
        self.histogram_widget.update_colormap(
            colormap_colors=np.asarray(layer.colormap.colors)[::-1],
            contrast_limits=[1.0 - limits[1], 1.0 - limits[0]],
            gamma=layer.gamma,
        )

    def _find_component_index_for_layer(self, layer):
        """Find which component index a layer belongs to based on its name."""
        layer_name = layer.name

        for i, comp in enumerate(self.components):
            if comp is not None:
                name = comp.name_edit.text().strip() or f"Component {i + 1}"

                # Check if layer name matches pattern for this component
                # Pattern: "{component_name} fractions: {source_layer}" or "{component_name} fraction: {source_layer}"
                if layer_name.startswith(
                    (f"{name} fractions: ", f"{name} fraction: ")
                ):
                    return i

        return None

    def _get_all_layers_for_component(self, comp_idx):
        """Get all fraction layers in the viewer belonging to a specific component for the current analysis mode."""
        if (
            comp_idx >= len(self.components)
            or self.components[comp_idx] is None
        ):
            return []

        comp = self.components[comp_idx]
        name = comp.name_edit.text().strip() or f"Component {comp_idx + 1}"

        if self.analysis_type == "Linear Projection":
            pattern = f"{name} fractions: "
        else:
            pattern = f"{name} fraction: "

        matching_layers = []
        for layer in self.viewer.layers:
            layer_name = layer.name
            if layer_name.startswith(pattern):
                matching_layers.append(layer)

        return matching_layers

    def _sync_component_layers_colormap(self, comp_idx, colormap):
        """Sync colormap across all layers belonging to the same component."""
        if getattr(self, '_updating_linked_layers', False):
            return

        try:
            self._updating_linked_layers = True
            layers_to_update = self._get_all_layers_for_component(comp_idx)

            for layer in layers_to_update:
                if layer.colormap != colormap:
                    layer.colormap = colormap
        finally:
            self._updating_linked_layers = False

    def _sync_component_layers_contrast_limits(
        self, comp_idx, contrast_limits
    ):
        """Sync contrast limits across all layers belonging to the same component."""
        if getattr(self, '_updating_linked_layers', False):
            return

        try:
            self._updating_linked_layers = True
            layers_to_update = self._get_all_layers_for_component(comp_idx)

            for layer in layers_to_update:
                if layer.contrast_limits != contrast_limits:
                    layer.contrast_limits = contrast_limits
        finally:
            self._updating_linked_layers = False

    def _sync_component_layers_gamma(self, comp_idx, gamma):
        """Sync gamma across all layers belonging to the same component."""
        if getattr(self, '_updating_linked_layers', False):
            return

        try:
            self._updating_linked_layers = True
            layers_to_update = self._get_all_layers_for_component(comp_idx)

            for layer in layers_to_update:
                if layer.gamma != gamma:
                    layer.gamma = gamma
        finally:
            self._updating_linked_layers = False

    def _is_standard_colormap(self, colormap_name):
        """Check if a colormap name refers to a standard matplotlib/vispy/napari colormap."""
        try:
            plt.get_cmap(colormap_name)
            return True
        except Exception:  # noqa: BLE001
            pass

        try:
            vispy.color.get_colormap(colormap_name)
            return True
        except Exception:  # noqa: BLE001
            pass

        try:
            from napari.utils.colormaps import AVAILABLE_COLORMAPS

            if colormap_name in AVAILABLE_COLORMAPS:
                return True
        except Exception:  # noqa: BLE001
            pass

        return False

    def _make_components_draggable(self):
        """Enable dragging of components and labels."""
        if self.drag_events_connected:
            return

        if self.parent_widget is None:
            return

        canvas = self.parent_widget.canvas_widget.canvas
        canvas.mpl_connect('button_press_event', self._on_press)
        canvas.mpl_connect('motion_notify_event', self._on_motion)
        canvas.mpl_connect('button_release_event', self._on_release)
        self.drag_events_connected = True

    def _on_press(self, event):
        """Handle press events for dragging components and labels."""
        if event.inaxes is None:
            return

        components_tab_is_active = (
            self.parent_widget is not None
            and getattr(self.parent_widget, "tab_widget", None) is not None
            and self.parent_widget.tab_widget.currentWidget()
            is self.parent_widget.components_tab
        )
        if not components_tab_is_active:
            return

        for comp in self.components:
            if comp.text is not None and comp.text.contains(event)[0]:
                if (
                    self.parent_widget.canvas_widget.toolbar.mode
                    == 'zoom rect'
                ):
                    with contextlib.suppress(Exception):
                        self.parent_widget.canvas_widget.toolbar.release_zoom(
                            event
                        )
                if self.parent_widget.canvas_widget.toolbar.mode == 'pan/zoom':
                    with contextlib.suppress(Exception):
                        self.parent_widget.canvas_widget.toolbar.release_pan(
                            event
                        )
                self.parent_widget.canvas_widget._on_escape(None)
                self.dragging_label_idx = comp.idx
                return
        for comp in self.components:
            if comp.dot is not None and comp.dot.contains(event)[0]:
                if (
                    self.parent_widget.canvas_widget.toolbar.mode
                    == 'zoom rect'
                ):
                    with contextlib.suppress(Exception):
                        self.parent_widget.canvas_widget.toolbar.release_zoom(
                            event
                        )
                if self.parent_widget.canvas_widget.toolbar.mode == 'pan/zoom':
                    with contextlib.suppress(Exception):
                        self.parent_widget.canvas_widget.toolbar.release_pan(
                            event
                        )
                self.parent_widget.canvas_widget._on_escape(None)
                self.dragging_component_idx = comp.idx
                return

    def _on_motion(self, event):
        """Handle dragging of components and labels."""
        if event.inaxes is None:
            return
        if self.dragging_label_idx is not None:
            comp = self.components[self.dragging_label_idx]
            if comp.text is not None and comp.dot is not None:
                x, y = event.xdata, event.ydata
                comp.text.set_position((x, y))
                dx, dy = comp.dot.get_data()
                comp.text_offset = (x - dx[0], y - dy[0])
                self.parent_widget.canvas_widget.canvas.draw_idle()
            return
        if self.dragging_component_idx is None:
            return
        comp = self.components[self.dragging_component_idx]
        if comp.dot is None:
            return
        x, y = event.xdata, event.ydata
        comp.dot.set_data([x], [y])
        comp.g_edit.setText(f"{x:.3f}")
        comp.s_edit.setText(f"{y:.3f}")

        if comp.lifetime_edit is not None:
            comp.lifetime_edit.clear()

        current_harmonic = getattr(self.parent_widget, 'harmonic', 1)
        lifetime = self._get_lifetime_from_coords(x, y, current_harmonic)

        if comp.lifetime_edit is not None:
            if lifetime is not None:
                comp.lifetime_edit.setText(f"{lifetime:.3f}")
            else:
                comp.lifetime_edit.clear()

        self._update_component_gs_coords(
            self.dragging_component_idx, current_harmonic, x, y
        )
        self._update_component_lifetime(
            self.dragging_component_idx, current_harmonic, lifetime
        )

        if comp.text is not None:
            ox, oy = comp.text_offset
            comp.text.set_position((x + ox, y + oy))
        self.draw_line_between_components()

    def _on_release(self, event):
        """Handle release of components and labels."""
        self.dragging_component_idx = None
        self.dragging_label_idx = None

    def _redraw(self, force=False):
        """Redraw the canvas. Use force=True to avoid stale blit artifacts."""
        if self.parent_widget is None:
            return
        canvas = self.parent_widget.canvas_widget.canvas
        if force and hasattr(canvas, "draw"):
            canvas.draw()
        else:
            canvas.draw_idle()

    def _get_harmonics_with_components(self):
        """Get list of harmonics that have component data from metadata."""
        harmonics = set()

        current_harmonic = getattr(self.parent_widget, 'harmonic', 1)
        active_components = [
            c for c in self.components if c is not None and c.dot is not None
        ]
        if len(active_components) > 0:
            harmonics.add(current_harmonic)

        if self.current_image_layer_name:
            # Check if layer still exists (defensive check for cleanup/teardown)
            if self.current_image_layer_name not in self.viewer.layers:
                return sorted(harmonics)

            layer = self.viewer.layers[self.current_image_layer_name]
            if (
                'settings' in layer.metadata
                and 'component_analysis' in layer.metadata['settings']
            ):

                settings = layer.metadata['settings']['component_analysis']
                components_data = settings.get('components', {})

                for comp_data in components_data.values():
                    gs_harmonics = comp_data.get('gs_harmonics', {})
                    for harmonic_str in gs_harmonics:
                        harmonics.add(int(harmonic_str))

        return sorted(harmonics)

    def _get_available_harmonics(self):
        """Get available harmonics from the phasor data."""
        if not self.current_image_layer_name:
            return []

        # Check if layer still exists (defensive check for cleanup/teardown)
        if self.current_image_layer_name not in self.viewer.layers:
            return []

        layer = self.viewer.layers[self.current_image_layer_name]
        harmonics = layer.metadata.get('harmonics')

        if harmonics is None:
            return []

        return sorted(np.atleast_1d(harmonics).tolist())

    def _get_next_harmonic(self, current_harmonic, available_harmonics):
        """Get the next available harmonic after the current one."""
        for harmonic in available_harmonics:
            if harmonic > current_harmonic:
                return harmonic
        return None

    def _get_inverted_colormap(self, colormap_or_name):
        """Get the inverted version of a colormap, creating it if necessary.

        Args:
            colormap_or_name: Either a colormap name (str) or a Colormap object
        """
        if isinstance(colormap_or_name, str):
            colormap_name = colormap_or_name
            colormap_colors = None
        elif isinstance(colormap_or_name, Colormap):
            colormap_name = getattr(colormap_or_name, 'name', 'custom')
            colormap_colors = getattr(colormap_or_name, 'colors', None)
        else:
            colormap_name = getattr(colormap_or_name, 'name', 'custom')
            colormap_colors = getattr(colormap_or_name, 'colors', None)

        if colormap_name.endswith('_r'):
            inverted_name = colormap_name[:-2]
        else:
            inverted_name = colormap_name + '_r'

        if self._is_standard_colormap(colormap_name):
            if self._is_standard_colormap(inverted_name):
                try:
                    mpl_cmap = plt.get_cmap(inverted_name)
                    colors = mpl_cmap(np.linspace(0, 1, 256))
                    return Colormap(colors=colors, name=inverted_name)
                except Exception:  # noqa: BLE001
                    if inverted_name in AVAILABLE_COLORMAPS:
                        return inverted_name

            try:
                base_name = (
                    colormap_name
                    if not colormap_name.endswith('_r')
                    else colormap_name[:-2]
                )
                mpl_cmap = plt.get_cmap(base_name)
                colors = mpl_cmap(np.linspace(0, 1, 256))
                inverted_colors = colors[::-1]
                return Colormap(
                    colors=inverted_colors, name=f"inverted_{base_name}"
                )
            except Exception:  # noqa: BLE001
                pass

        if colormap_colors is not None:
            if isinstance(colormap_colors, list):
                colormap_colors = np.array(colormap_colors)
            inverted_colors = colormap_colors[::-1]
            return Colormap(
                colors=inverted_colors, name=f"inverted_{colormap_name}"
            )

        return 'jet_r' if not colormap_name.endswith('_r') else 'jet'

    def _find_and_reconnect_layer(
        self, expected_name, component_name, layer_name, idx
    ):
        """Find and reconnect to an existing fraction layer by various naming conventions."""
        if expected_name in self.viewer.layers:
            if idx == 0:
                self.comp1_fractions_layer = self.viewer.layers[expected_name]
                self.comp1_fractions_layer.events.colormap.connect(
                    self._on_colormap_changed
                )
                self.comp1_fractions_layer.events.contrast_limits.connect(
                    self._on_contrast_limits_changed
                )
                self.comp1_fractions_layer.events.gamma.connect(
                    self._on_colormap_changed
                )
        else:
            possible_names = [
                f"Component {idx + 1} fractions: {layer_name}",
                f"{component_name} fractions: {layer_name}",
            ]

            for possible_name in possible_names:
                if possible_name in self.viewer.layers:
                    layer_obj = self.viewer.layers[possible_name]
                    layer_obj.name = expected_name

                    if idx == 0:
                        self.comp1_fractions_layer = layer_obj
                        self.comp1_fractions_layer.events.colormap.connect(
                            self._on_colormap_changed
                        )
                        self.comp1_fractions_layer.events.contrast_limits.connect(
                            self._on_contrast_limits_changed
                        )
                        self.comp1_fractions_layer.events.gamma.connect(
                            self._on_colormap_changed
                        )
                    break

    def _reconnect_existing_fraction_layers(self, layer_name):
        """Reconnect to existing fraction layers if they exist."""
        layer = self.viewer.layers[layer_name]

        comp1_name = "Component 1"

        if (
            'settings' in layer.metadata
            and 'component_analysis' in layer.metadata['settings']
        ):

            settings = layer.metadata['settings']['component_analysis']

            if (
                'components' in settings
                and len(settings['components']) > 0
                and '0' in settings['components']
            ):
                comp1_name = (
                    settings['components']['0'].get('name') or "Component 1"
                )

        comp1_fractions_layer_name = f"{comp1_name} fractions: {layer_name}"

        self._find_and_reconnect_layer(
            comp1_fractions_layer_name, comp1_name, layer_name, 0
        )

        if self.comp1_fractions_layer is not None:
            self.fractions_colormap = (
                self.comp1_fractions_layer.colormap.colors
            )
            self.colormap_contrast_limits = (
                self.comp1_fractions_layer.contrast_limits
            )

    def _on_image_layer_changed(self):
        """Callback whenever the image layer with phasor features changes."""
        self._teardown_on_layer_change()
        self._restore_on_layer_change()
        self._refresh_run_button_if_ready()

    def _teardown_on_layer_change(self):
        """Immediate cleanup: remove artists and disconnect signals."""
        self.current_image_layer_name = (
            self.parent_widget.get_primary_layer_name()
        )

        for comp in self.components:
            if comp is not None:
                if comp.dot is not None:
                    comp.dot.remove()
                    comp.dot = None
                if comp.text is not None:
                    comp.text.remove()
                    comp.text = None

        if self.component_line is not None:
            with contextlib.suppress(ValueError, AttributeError):
                self.component_line.remove()
            self.component_line = None

        if self.component_polygon is not None:
            with contextlib.suppress(ValueError, AttributeError):
                self.component_polygon.remove()
            self.component_polygon = None

        if self.comp1_fractions_layer is not None:
            try:
                self.comp1_fractions_layer.events.colormap.disconnect(
                    self._on_colormap_changed
                )
                self.comp1_fractions_layer.events.contrast_limits.disconnect(
                    self._on_contrast_limits_changed
                )
                self.comp1_fractions_layer.events.gamma.disconnect(
                    self._on_colormap_changed
                )
            except Exception:  # noqa: BLE001
                pass

        self.comp1_fractions_layer = None
        self.comp2_fractions_layer = None
        self.fractions_colormap = None
        self.colormap_contrast_limits = None
        self.fractions_gamma = 1.0

    def _restore_on_layer_change(self):
        """Deferred restore: update UI state from metadata."""
        self._needs_update = False

        self._update_lifetime_inputs_visibility()

        layer_name = self.parent_widget.get_primary_layer_name()
        if layer_name:
            self._reconnect_existing_fraction_layers(layer_name)

            self._restore_components_ui_only_from_metadata()

        else:
            self._updating_settings = True
            try:
                for comp in self.components:
                    if comp is not None:
                        comp.g_edit.clear()
                        comp.s_edit.clear()
                        comp.name_edit.clear()
                        if comp.lifetime_edit is not None:
                            comp.lifetime_edit.clear()
            finally:
                self._updating_settings = False

    def _ensure_component_metadata(self, idx: int, harmonic: int = None):
        """Ensure component metadata structure exists and return component data dict."""
        if self._updating_settings or not self.current_image_layer_name:
            return None

        # Check if layer still exists (defensive check for cleanup/teardown)
        if self.current_image_layer_name not in self.viewer.layers:
            return None

        layer = self.viewer.layers[self.current_image_layer_name]
        if 'settings' not in layer.metadata:
            layer.metadata['settings'] = {}
        if 'component_analysis' not in layer.metadata['settings']:
            layer.metadata['settings'][
                'component_analysis'
            ] = self._get_default_components_settings()

        settings = layer.metadata['settings']['component_analysis']
        idx_str = str(idx)

        if idx_str not in settings['components']:
            comp = self.components[idx]
            settings['components'][idx_str] = {
                'idx': idx,
                'name': comp.name_edit.text().strip() or None,
                'gs_harmonics': {},
            }

        if harmonic is not None:
            harmonic_key = str(harmonic)
            if (
                harmonic_key
                not in settings['components'][idx_str]['gs_harmonics']
            ):
                settings['components'][idx_str]['gs_harmonics'][
                    harmonic_key
                ] = {}

        return settings['components'][idx_str]

    def _update_component_gs_coords(
        self, idx: int, harmonic: int, g: float, s: float
    ):
        """Update component G/S coordinates for a specific harmonic."""
        comp_data = self._ensure_component_metadata(idx, harmonic)
        if comp_data is None:
            return

        harmonic_key = str(harmonic)
        comp_data['gs_harmonics'][harmonic_key]['g'] = g
        comp_data['gs_harmonics'][harmonic_key]['s'] = s

    def _update_component_lifetime(
        self, idx: int, harmonic: int, lifetime: float
    ):
        """Update component lifetime for a specific harmonic."""
        comp_data = self._ensure_component_metadata(idx, harmonic)
        if comp_data is None:
            return

        harmonic_key = str(harmonic)
        comp_data['gs_harmonics'][harmonic_key]['lifetime'] = lifetime

    def _update_component_name(self, idx: int, name: str):
        """Update component name."""
        comp_data = self._ensure_component_metadata(idx)
        if comp_data is None:
            return

        comp_data['name'] = name if name else None

    def _update_component_phasor_center_layers(
        self, idx: int, layer_names: list[str]
    ):
        """Persist the layers used to compute this component's phasor center."""
        comp_data = self._ensure_component_metadata(idx)
        if comp_data is None:
            return

        comp_data['phasor_center_layers'] = list(layer_names)

    def _update_component_colormap(
        self,
        idx: int,
        harmonic: int,
        colormap_name: str,
        colormap_colors: list,
        contrast_limits: tuple,
    ):
        """Update component colormap settings for a specific harmonic."""
        comp_data = self._ensure_component_metadata(idx, harmonic)
        if comp_data is None:
            return

        harmonic_key = str(harmonic)
        comp_data['gs_harmonics'][harmonic_key].update(
            {
                'colormap_name': colormap_name,
                'colormap_colors': colormap_colors,
                'contrast_limits': (
                    list(contrast_limits) if contrast_limits else None
                ),
                'analysis_type': self.analysis_type,  # Store which analysis type saved this
            }
        )

    def _run_analysis(self):
        """Run the selected analysis and store component locations in metadata for all selected layers."""
        self._analysis_attempted = True
        self._update_all_component_styling()

        selected_layers = self.parent_widget.get_selected_layers()
        if not self._updating_settings and selected_layers:
            # Store metadata in all selected layers
            for layer in selected_layers:
                if 'settings' not in layer.metadata:
                    layer.metadata['settings'] = {}
                if 'component_analysis' not in layer.metadata['settings']:
                    layer.metadata['settings'][
                        'component_analysis'
                    ] = self._get_default_components_settings()

                settings = layer.metadata['settings']['component_analysis']

                if 'components' not in settings:
                    settings['components'] = {}

                current_harmonic = getattr(self.parent_widget, 'harmonic', 1)
                settings['last_analysis_harmonic'] = current_harmonic

                active_components = [
                    c
                    for c in self.components
                    if c is not None and c.dot is not None
                ]

                for comp in active_components:
                    idx = comp.idx
                    idx_str = str(idx)
                    name = comp.name_edit.text().strip()

                    if idx_str not in settings['components']:
                        settings['components'][idx_str] = {
                            'idx': idx,
                            'name': name if name else None,
                            'gs_harmonics': {},
                        }
                    else:
                        if name:
                            settings['components'][idx_str]['name'] = name

                    comp_data = settings['components'][idx_str]

                    if 'gs_harmonics' not in comp_data:
                        comp_data['gs_harmonics'] = {}

                    try:
                        x_data, y_data = comp.dot.get_data()
                        g_val = x_data[0]
                        s_val = y_data[0]
                    except (ValueError, IndexError):
                        continue

                    harmonic_key = str(current_harmonic)
                    if harmonic_key not in comp_data['gs_harmonics']:
                        comp_data['gs_harmonics'][harmonic_key] = {}

                    comp_data['gs_harmonics'][harmonic_key]['g'] = g_val
                    comp_data['gs_harmonics'][harmonic_key]['s'] = s_val

                    if comp.lifetime_edit is not None:
                        lifetime_text = comp.lifetime_edit.text().strip()
                        if lifetime_text:
                            try:
                                lifetime_val = float(lifetime_text)
                                comp_data['gs_harmonics'][harmonic_key][
                                    'lifetime'
                                ] = lifetime_val
                            except ValueError:
                                comp_data['gs_harmonics'][harmonic_key][
                                    'lifetime'
                                ] = None
                    else:
                        comp_data['gs_harmonics'][harmonic_key][
                            'lifetime'
                        ] = None

                settings['analysis_type'] = self.analysis_type

        if self.analysis_type == "Linear Projection":
            self._run_linear_projection()
        else:
            self._run_component_fit()

        self._update_histogram_combobox()
        self.update_component_histogram()

    def _run_linear_projection(self):
        """Run linear projection for 2-component analysis on all selected layers."""
        selected_layers = self.parent_widget.get_selected_layers()
        if not selected_layers:
            return
        if not all(c.dot is not None for c in self.components[:2]):
            return

        c1, c2 = self.components[:2]
        component_real = (c1.dot.get_data()[0][0], c2.dot.get_data()[0][0])
        component_imag = (c1.dot.get_data()[1][0], c2.dot.get_data()[1][0])

        for layer in selected_layers:
            self._run_linear_projection_for_layer(
                layer, component_real, component_imag, c1, c2
            )

    def _run_linear_projection_for_layer(
        self, layer, component_real, component_imag, c1, c2
    ):
        """Run linear projection for a single layer."""
        g_array = layer.metadata.get('G')
        s_array = layer.metadata.get('S')
        harmonics = layer.metadata.get('harmonics')

        if g_array is None or s_array is None:
            return

        if harmonics is not None:
            try:
                harmonics_array = np.atleast_1d(harmonics)
                harmonic_idx = np.where(
                    harmonics_array == self.parent_widget.harmonic
                )[0][0]
                if g_array.ndim == layer.data.ndim + 1:
                    real = g_array[harmonic_idx]
                    imag = s_array[harmonic_idx]
                else:
                    real = g_array
                    imag = s_array
            except IndexError:
                return
        else:
            real = g_array
            imag = s_array

        fraction_comp1 = phasor_component_fraction(
            real, imag, component_real, component_imag
        )

        comp1_name = c1.name_edit.text().strip() or "Component 1"
        comp1_fractions_layer_name = f"{comp1_name} fractions: {layer.name}"

        settings = layer.metadata.get('settings', {}).get(
            'component_analysis', {}
        )
        current_harmonic = getattr(self.parent_widget, 'harmonic', 1)
        harmonic_key = str(current_harmonic)

        comp1_colormap = None
        comp1_gamma = None
        valid_fraction = fraction_comp1[np.isfinite(fraction_comp1)]
        if valid_fraction.size > 0:
            frac_min = float(np.min(valid_fraction))
            frac_max = float(np.max(valid_fraction))
            if frac_max <= frac_min:
                frac_max = frac_min + 0.01
            contrast_limits = (frac_min, frac_max)
        else:
            contrast_limits = (0, 1)

        # If the fraction layer is already displayed (the user clicked the
        # button again), preserve its current colormap, contrast limits and
        # gamma. This takes precedence over the stored defaults so manual
        # display tweaks - including colormaps propagated from other analyzed
        # images - survive a re-display.
        if comp1_fractions_layer_name in self.viewer.layers:
            existing_layer = self.viewer.layers[comp1_fractions_layer_name]
            comp1_colormap = existing_layer.colormap
            contrast_limits = existing_layer.contrast_limits
            comp1_gamma = existing_layer.gamma

        if comp1_colormap is None and '0' in settings.get('components', {}):
            comp_data = settings['components']['0']
            if harmonic_key in comp_data.get('gs_harmonics', {}):
                harmonic_data = comp_data['gs_harmonics'][harmonic_key]

                saved_analysis_type = harmonic_data.get('analysis_type')
                if saved_analysis_type == 'Linear Projection':
                    has_saved_colormap = (
                        harmonic_data.get('colormap_colors') is not None
                        or harmonic_data.get('colormap_name') is not None
                    )

                    if has_saved_colormap:
                        if harmonic_data.get('colormap_colors') is not None:
                            colors = harmonic_data['colormap_colors']
                            if isinstance(colors, list):
                                colors = np.array(colors)

                            stored_colormap_name = harmonic_data.get(
                                'colormap_name', 'custom'
                            )
                            comp1_colormap = Colormap(
                                colors=colors,
                                name=(
                                    stored_colormap_name
                                    if stored_colormap_name
                                    else 'custom'
                                ),
                            )

                        elif harmonic_data.get('colormap_name'):
                            comp1_colormap = harmonic_data['colormap_name']

                        if harmonic_data.get('contrast_limits'):
                            contrast_limits = tuple(
                                harmonic_data['contrast_limits']
                            )
                        if harmonic_data.get('gamma') is not None:
                            comp1_gamma = harmonic_data['gamma']

        if comp1_colormap is None:
            comp1_colormap = 'jet'

        if comp1_fractions_layer_name in self.viewer.layers:
            self.viewer.layers.remove(
                self.viewer.layers[comp1_fractions_layer_name]
            )

        comp1_selected_fractions_layer = Image(
            fraction_comp1,
            name=comp1_fractions_layer_name,
            scale=layer.scale,
            colormap=comp1_colormap,
            contrast_limits=contrast_limits,
        )

        self.comp1_fractions_layer = self.viewer.add_layer(
            comp1_selected_fractions_layer
        )
        if comp1_gamma is not None:
            self.comp1_fractions_layer.gamma = comp1_gamma
        self.comp1_fractions_layer.metadata['fraction_data_original'] = (
            fraction_comp1.copy()
        )

        self.fractions_colormap = self.comp1_fractions_layer.colormap.colors
        self.colormap_contrast_limits = (
            self.comp1_fractions_layer.contrast_limits
        )

        if not self._updating_settings and self.current_image_layer_name:
            colormap_name = getattr(
                self.comp1_fractions_layer.colormap, 'name', 'custom'
            )
            is_standard = self._is_standard_colormap(colormap_name)

            if colormap_name.startswith('inverted_'):
                is_standard = False

            if '0' in settings['components']:
                comp_data = settings['components']['0']
                if harmonic_key not in comp_data['gs_harmonics']:
                    comp_data['gs_harmonics'][harmonic_key] = {}

                if is_standard:
                    comp_data['gs_harmonics'][harmonic_key][
                        'colormap_name'
                    ] = colormap_name
                    comp_data['gs_harmonics'][harmonic_key][
                        'colormap_colors'
                    ] = None
                else:
                    colormap_colors = (
                        self.comp1_fractions_layer.colormap.colors
                    )
                    if colormap_colors is not None and (
                        hasattr(colormap_colors, 'tolist')
                        or isinstance(colormap_colors, np.ndarray)
                    ):
                        colormap_colors = colormap_colors.tolist()
                    comp_data['gs_harmonics'][harmonic_key][
                        'colormap_name'
                    ] = colormap_name
                    comp_data['gs_harmonics'][harmonic_key][
                        'colormap_colors'
                    ] = colormap_colors

                comp_data['gs_harmonics'][harmonic_key]['contrast_limits'] = (
                    list(self.comp1_fractions_layer.contrast_limits)
                )
                comp_data['gs_harmonics'][harmonic_key][
                    'gamma'
                ] = self.comp1_fractions_layer.gamma

        self.comp1_fractions_layer.events.colormap.connect(
            self._on_colormap_changed
        )
        self.comp1_fractions_layer.events.contrast_limits.connect(
            self._on_contrast_limits_changed
        )
        self.comp1_fractions_layer.events.gamma.connect(
            self._on_colormap_changed
        )

        self._update_component_colors()
        self.draw_line_between_components()

    def _run_component_fit(self):
        """Run multi-component analysis using phasor_component_fit on all selected layers."""
        selected_layers = self.parent_widget.get_selected_layers()
        if not selected_layers:
            return

        active_components = [
            c for c in self.components if c is not None and c.dot is not None
        ]
        if len(active_components) < 2:
            return

        num_components = len(active_components)
        current_harmonic = getattr(self.parent_widget, 'harmonic', 1)

        required_harmonics = self._get_required_harmonics(num_components)

        if required_harmonics > 1:
            available_harmonics = self._get_available_harmonics()
            if len(available_harmonics) < required_harmonics:
                show_warning(
                    f"{num_components}-component analysis requires at least "
                    f"{required_harmonics} harmonics in the data"
                )
                return

        for layer in selected_layers:
            self._run_component_fit_for_layer(
                layer,
                active_components,
                num_components,
                current_harmonic,
                required_harmonics,
            )

    def _run_component_fit_for_layer(
        self,
        layer,
        active_components,
        num_components,
        current_harmonic,
        required_harmonics,
    ):
        """Run component fit analysis for a single layer."""
        if required_harmonics > 1:
            available_harmonics = self._get_available_harmonics()
            harmonics_with_components = self._get_harmonics_with_components()
            if len(harmonics_with_components) < required_harmonics:

                next_harmonic = self._get_next_harmonic(
                    current_harmonic, available_harmonics
                )
                if next_harmonic is not None:
                    self.parent_widget.harmonic = next_harmonic

                    if hasattr(self.parent_widget, 'harmonic_spinbox'):
                        self.parent_widget.harmonic_spinbox.setValue(
                            next_harmonic
                        )
                    show_info(
                        f"For {num_components}-component analysis, please "
                        "select component locations in harmonic "
                        f"{next_harmonic} as well"
                    )
                    return
                else:
                    show_info(
                        f"{num_components}-component analysis requires"
                        " component locations in at least "
                        f"{required_harmonics} harmonics"
                    )
                    return

            if len(harmonics_with_components) > required_harmonics:
                harmonics_with_components = harmonics_with_components[
                    :required_harmonics
                ]

        g_array = layer.metadata.get('G')
        s_array = layer.metadata.get('S')
        harmonics = layer.metadata.get('harmonics')
        mean = layer.metadata.get('original_mean')

        if harmonics is not None:
            harmonics = np.atleast_1d(harmonics)

        if required_harmonics == 1:
            component_g, component_s, component_names = (
                self._get_component_coords_for_harmonic(current_harmonic)
            )
            if not component_g:
                show_warning(
                    f"No components found for harmonic {current_harmonic}"
                )
                return

            if harmonics is not None:
                try:
                    harmonic_idx = np.where(harmonics == current_harmonic)[0][
                        0
                    ]
                    if g_array.ndim == layer.data.ndim + 1:
                        real = g_array[harmonic_idx]
                        imag = s_array[harmonic_idx]
                    else:
                        real = g_array
                        imag = s_array
                except IndexError:
                    return
            else:
                real = g_array
                imag = s_array

        else:
            harmonics_with_components = sorted(
                self._get_harmonics_with_components()
            )

            if len(harmonics_with_components) > required_harmonics:
                harmonics_with_components = harmonics_with_components[
                    :required_harmonics
                ]

            component_counts = []
            for harmonic in harmonics_with_components:
                g_coords, s_coords, names = (
                    self._get_component_coords_for_harmonic(harmonic)
                )
                component_counts.append(len(g_coords))

            if not all(count == num_components for count in component_counts):
                show_error(
                    f"All harmonics must have exactly {num_components} "
                    f"component locations. Found: {component_counts} "
                    f"components in harmonics {harmonics_with_components}."
                )
                return

            component_g = []
            component_s = []
            component_names = []

            for harmonic in harmonics_with_components:
                g_coords, s_coords, names = (
                    self._get_component_coords_for_harmonic(harmonic)
                )
                component_g.append(g_coords)
                component_s.append(s_coords)
                if not component_names:
                    component_names = [
                        name.split('_H')[0] if '_H' in name else name
                        for name in names
                    ]

            real_list = []
            imag_list = []

            for harmonic in harmonics_with_components:
                try:
                    harmonic_idx = np.where(harmonics == harmonic)[0][0]
                    if g_array.ndim == layer.data.ndim + 1:
                        real_list.append(g_array[harmonic_idx])
                        imag_list.append(s_array[harmonic_idx])
                    else:
                        real_list.append(g_array)
                        imag_list.append(s_array)
                except IndexError:
                    continue

            real = np.stack(real_list, axis=0)
            imag = np.stack(imag_list, axis=0)

        try:
            fractions = phasor_component_fit(
                mean, real, imag, component_g, component_s
            )

            settings = layer.metadata['settings']['component_analysis']
            harmonic_key = str(current_harmonic)

            self.fraction_layers.clear()

            for i, (fraction, name) in enumerate(
                zip(fractions, component_names, strict=False)
            ):
                fraction_layer_name = f"{name} fraction: {layer.name}"

                colormap = None
                gamma = None
                valid_fraction = fraction[np.isfinite(fraction)]
                if valid_fraction.size > 0:
                    frac_min = float(np.min(valid_fraction))
                    frac_max = float(np.max(valid_fraction))
                    if frac_max <= frac_min:
                        frac_max = frac_min + 0.01
                    contrast_limits = (frac_min, frac_max)
                else:
                    contrast_limits = (0, 1)
                idx_str = str(i)

                # If the fraction layer is already displayed (the user clicked
                # the button again), preserve its current colormap, contrast
                # limits and gamma. This takes precedence over the stored
                # defaults so manual display tweaks - including colormaps
                # propagated from other analyzed images - survive a re-display.
                if fraction_layer_name in self.viewer.layers:
                    existing_layer = self.viewer.layers[fraction_layer_name]
                    colormap = existing_layer.colormap
                    contrast_limits = existing_layer.contrast_limits
                    gamma = existing_layer.gamma

                if colormap is None and (
                    not self._updating_settings
                    and idx_str in settings.get('components', {})
                    and harmonic_key
                    in settings['components'][idx_str].get('gs_harmonics', {})
                ):

                    harmonic_data = settings['components'][idx_str][
                        'gs_harmonics'
                    ][harmonic_key]

                    saved_analysis_type = harmonic_data.get('analysis_type')
                    if saved_analysis_type == 'Component Fit':
                        if harmonic_data.get('colormap_name'):
                            colormap = harmonic_data['colormap_name']
                        elif harmonic_data.get('colormap_colors'):
                            from napari.utils.colormaps import Colormap

                            colors = harmonic_data['colormap_colors']
                            if isinstance(colors, list):
                                colors = np.array(colors)
                            colormap = Colormap(
                                colors=colors, name="saved_custom"
                            )

                        if harmonic_data.get('contrast_limits'):
                            contrast_limits = tuple(
                                harmonic_data['contrast_limits']
                            )
                        if harmonic_data.get('gamma') is not None:
                            gamma = harmonic_data['gamma']

                if colormap is None:
                    if i < len(self.component_colormap_names):
                        colormap = self.component_colormap_names[i]
                    else:
                        colormap = 'viridis'

                with contextlib.suppress(KeyError):
                    self.viewer.layers.remove(
                        self.viewer.layers[fraction_layer_name]
                    )
                new_layer = self.viewer.add_image(
                    fraction,
                    name=fraction_layer_name,
                    scale=layer.scale,
                    colormap=colormap,
                )
                new_layer.metadata['fraction_data_original'] = fraction.copy()

                new_layer.contrast_limits = contrast_limits
                if gamma is not None:
                    new_layer.gamma = gamma

                self.fraction_layers.append(new_layer)
                new_layer.events.colormap.connect(self._on_colormap_changed)
                new_layer.events.contrast_limits.connect(
                    self._on_contrast_limits_changed
                )
                new_layer.events.gamma.connect(self._on_colormap_changed)

                if (
                    not self._updating_settings
                    and idx_str in settings['components']
                ):
                    comp_data = settings['components'][idx_str]
                    if harmonic_key not in comp_data['gs_harmonics']:
                        comp_data['gs_harmonics'][harmonic_key] = {}

                    colormap_name = getattr(
                        new_layer.colormap, 'name', 'custom'
                    )
                    is_standard_colormap = False
                    try:
                        plt.get_cmap(colormap_name)
                        is_standard_colormap = True
                    except Exception:  # noqa: BLE001
                        try:
                            vispy.color.get_colormap(colormap_name)
                            is_standard_colormap = True
                        except Exception:  # noqa: BLE001
                            is_standard_colormap = False

                    if is_standard_colormap:
                        comp_data['gs_harmonics'][harmonic_key][
                            'colormap_name'
                        ] = colormap_name
                        comp_data['gs_harmonics'][harmonic_key][
                            'colormap_colors'
                        ] = None
                    else:
                        colormap_colors = new_layer.colormap.colors
                        if colormap_colors is not None and (
                            hasattr(colormap_colors, 'tolist')
                            or isinstance(colormap_colors, np.ndarray)
                        ):
                            colormap_colors = colormap_colors.tolist()
                        comp_data['gs_harmonics'][harmonic_key][
                            'colormap_name'
                        ] = None
                        comp_data['gs_harmonics'][harmonic_key][
                            'colormap_colors'
                        ] = colormap_colors

                    comp_data['gs_harmonics'][harmonic_key][
                        'contrast_limits'
                    ] = list(new_layer.contrast_limits)
                    comp_data['gs_harmonics'][harmonic_key][
                        'gamma'
                    ] = new_layer.gamma

            self._update_component_colors()

        except Exception as e:  # noqa: BLE001
            show_error(f"Analysis failed: {str(e)}")

    def rename_layer(self, old_name: str, new_name: str):
        """Rename derived layers when base layer is renamed."""
        for layer in self.viewer.layers:
            if not isinstance(layer, Image):
                continue
            name = layer.name
            for sep in (" fractions: ", " fraction: "):
                if name.endswith(sep + old_name):
                    comp_part = name[: -len(sep + old_name)]
                    layer.name = f"{comp_part}{sep}{new_name}"

        # Keep each component's remembered phasor-center selection in sync so a
        # renamed layer stays selected instead of being dropped.
        for comp in self.components:
            if comp is None or not comp.phasor_center_layers:
                continue
            if old_name in comp.phasor_center_layers:
                comp.phasor_center_layers = [
                    new_name if n == old_name else n
                    for n in comp.phasor_center_layers
                ]
                self._update_component_phasor_center_layers(
                    comp.idx, comp.phasor_center_layers
                )

    def _get_fraction_layers_for_component(self, component_name):
        """Get all fraction layers in the viewer for a given component name.

        Searches the viewer for fraction layers matching the component name
        pattern, regardless of which image layer they belong to.

        Parameters
        ----------
        component_name : str
            The component display name to search for.

        Returns
        -------
        dict
            ``{image_layer_name: napari.layers.Image}`` mapping each source
            image layer name to its fraction layer.
        """
        result = {}
        # Linear projection: "<comp_name> fractions: <layer_name>"
        # Component fit:     "<comp_name> fraction: <layer_name>"
        for layer in self.viewer.layers:
            if not isinstance(layer, Image):
                continue
            name = layer.name
            for sep in (" fractions: ", " fraction: "):
                if name.startswith(component_name + sep):
                    img_layer_name = name[len(component_name) + len(sep) :]
                    result[img_layer_name] = layer
                    break
        return result

    def _get_component_names_from_fraction_layers(self):
        """Discover unique component names from fraction layers in the viewer.

        Uses custom names from metadata if available, otherwise falls back
        to names extracted from layer names.

        Returns
        -------
        list of str
            Unique component names found in the viewer's fraction layers.
        """
        layer_based_names = {}
        for layer in self.viewer.layers:
            if not isinstance(layer, Image):
                continue
            for sep in (" fractions: ", " fraction: "):
                idx = layer.name.find(sep)
                if idx != -1:
                    comp_name = layer.name[:idx]
                    if comp_name.startswith("Component "):
                        try:
                            comp_idx = int(comp_name.split(" ")[1]) - 1
                            layer_based_names[comp_idx] = comp_name
                        except (ValueError, IndexError):
                            layer_based_names[comp_name] = comp_name
                    else:
                        layer_based_names[comp_name] = comp_name
                    break

        if not layer_based_names:
            return []

        layer_name = (
            self.parent_widget.get_primary_layer_name()
            if self.parent_widget
            else None
        )
        if layer_name and layer_name in self.viewer.layers:
            layer = self.viewer.layers[layer_name]
            settings = layer.metadata.get('settings', {}).get(
                'component_analysis', {}
            )
            components_data = settings.get('components', {})

            for idx_str, comp_data in components_data.items():
                try:
                    idx = int(idx_str)
                    custom_name = comp_data.get('name')
                    if custom_name and idx in layer_based_names:
                        del layer_based_names[idx]
                        layer_based_names[custom_name] = custom_name
                except (ValueError, KeyError):
                    continue

        seen = set()
        result = []
        for name in layer_based_names.values():
            if name not in seen:
                seen.add(name)
                result.append(name)
        return result

    def _linear_projection_component_names(self):
        """First/second component display names for Linear Projection mode.

        In Linear Projection only the first component gets a fraction layer.
        The second component's fraction is ``1 - first`` and can be displayed
        in the histogram without creating an extra layer.

        Returns
        -------
        tuple of (str or None, str or None)
            ``(name1, name2)`` when applicable, otherwise ``(None, None)``.
        """
        if self.analysis_type != "Linear Projection":
            return None, None
        if len(self.components) < 2:
            return None, None
        c1, c2 = self.components[0], self.components[1]
        if c1 is None or c2 is None:
            return None, None
        name1 = c1.name_edit.text().strip() or "Component 1"
        name2 = c2.name_edit.text().strip() or "Component 2"
        if name1 == name2:
            return None, None
        return name1, name2

    def _resolve_histogram_component(self, selected_text):
        """Resolve a combobox entry to its fraction layers and an invert flag.

        Most entries map directly to fraction layers in the viewer. In Linear
        Projection mode the second component has no layer of its own; its
        fraction is ``1 - first_component_fraction``. In that case the
        underlying first-component layers are returned together with
        ``invert=True`` so the displayed values, colormap and contrast limits
        are inverted.

        Returns
        -------
        tuple of (dict, bool)
            ``({image_layer_name: Image}, invert)``.
        """
        fraction_layers_map = self._get_fraction_layers_for_component(
            selected_text
        )
        if fraction_layers_map:
            return fraction_layers_map, False

        name1, name2 = self._linear_projection_component_names()
        if name2 is not None and selected_text == name2:
            comp1_layers = self._get_fraction_layers_for_component(name1)
            if comp1_layers:
                return comp1_layers, True

        return {}, False

    def _update_histogram_combobox(self):
        """Populate the component selector comboboxes with fraction layers.

        The histogram and statistics docks each show a combobox; both are
        populated with the same entries and kept in sync.
        """
        current_text = self.histogram_component_combobox.currentText()

        comp_names = list(self._get_component_names_from_fraction_layers())

        # Linear Projection creates only the first component's fraction layer,
        # but the second component's fraction (1 - first) can also be shown.
        name1, name2 = self._linear_projection_component_names()
        if (
            name2 is not None
            and self._get_fraction_layers_for_component(name1)
            and name2 not in comp_names
        ):
            comp_names.append(name2)

        for combobox in (
            self.histogram_component_combobox,
            self.stats_component_combobox,
        ):
            combobox.blockSignals(True)
            combobox.clear()
            for comp_name in comp_names:
                combobox.addItem(comp_name)
            idx = combobox.findText(current_text)
            if idx >= 0:
                combobox.setCurrentIndex(idx)
            combobox.blockSignals(False)

    def _on_histogram_component_changed(self, index):
        """Handle change of the selected component in the histogram combobox."""
        self._mirror_component_selection(
            self.histogram_component_combobox, self.stats_component_combobox
        )
        self.update_component_histogram()

    def _on_stats_component_changed(self, index):
        """Handle change of the selected component in the statistics combobox.

        Mirrors the choice onto the histogram combobox, which is the canonical
        selector and refreshes the histogram/statistics via its own signal.
        """
        self._mirror_component_selection(
            self.stats_component_combobox, self.histogram_component_combobox
        )

    def _mirror_component_selection(self, source, target):
        """Copy the current selection from ``source`` to ``target`` combobox."""
        if self._syncing_component_comboboxes:
            return
        text = source.currentText()
        if target.currentText() == text:
            return
        idx = target.findText(text)
        if idx < 0:
            return
        self._syncing_component_comboboxes = True
        try:
            target.setCurrentIndex(idx)
        finally:
            self._syncing_component_comboboxes = False

    def _on_fraction_range_changed(self, min_val, max_val):
        """Handle range slider changes on the fraction histogram.

        Clips fraction layers to the specified range and refreshes the
        histogram accordingly.
        """
        selected_text = self.histogram_component_combobox.currentText()
        if not selected_text:
            return

        fraction_layers_map, invert = self._resolve_histogram_component(
            selected_text
        )
        if not fraction_layers_map:
            return

        # When the second component is displayed, the slider acts on its
        # fraction (1 - first). Clipping it to [min_val, max_val] is equivalent
        # to clipping the underlying first-component layer to [1-max, 1-min].
        if invert:
            layer_min, layer_max = 1.0 - max_val, 1.0 - min_val
        else:
            layer_min, layer_max = min_val, max_val

        clipped_data = {}
        self._updating_linked_layers = True
        try:
            for img_name, fl in fraction_layers_map.items():
                # Use original (unclipped) values when available so expanding the
                # slider range can restore previous data.
                original = fl.metadata.get('fraction_data_original', fl.data)
                clipped = np.clip(original, layer_min, layer_max)
                fl.data = clipped

                current_limits = np.asarray(fl.contrast_limits, dtype=float)
                target_limits = np.asarray([layer_min, layer_max], dtype=float)
                if not np.allclose(current_limits, target_limits):
                    fl.contrast_limits = [layer_min, layer_max]

                clipped_data[img_name] = 1.0 - clipped if invert else clipped
        finally:
            self._updating_linked_layers = False

        # The phasor-plot line gradient is always expressed in first-component
        # fraction space, so keep ``colormap_contrast_limits`` in that space.
        self.colormap_contrast_limits = [layer_min, layer_max]
        first_layer = next(iter(fraction_layers_map.values()))
        colormap_colors = first_layer.colormap.colors
        if invert:
            colormap_colors = np.asarray(colormap_colors)[::-1]
        self.histogram_widget.update_colormap(
            colormap_colors=colormap_colors,
            contrast_limits=[min_val, max_val],
            gamma=first_layer.gamma,
        )

        # Pool the (clipped) data from every selected layer into a single
        # merged histogram so all layers contribute, rather than showing a
        # per-layer mean +/- SD that visually resembles a single layer.
        if len(clipped_data) > 1:
            pooled = np.concatenate(
                [
                    np.asarray(d, dtype=float).ravel()
                    for d in clipped_data.values()
                ]
            )
            self.histogram_widget.update_data(pooled)
        else:
            first_data = next(iter(clipped_data.values()))
            self.histogram_widget.update_data(first_data)

        self.draw_line_between_components()

    def update_component_histogram(self):
        """Update the histogram with the fraction data of the selected component."""
        selected_text = self.histogram_component_combobox.currentText()
        if not selected_text:
            self.histogram_widget.hide()
            return

        fraction_layers_map, invert = self._resolve_histogram_component(
            selected_text
        )
        if not fraction_layers_map:
            self.histogram_widget.hide()
            return

        first_layer = next(iter(fraction_layers_map.values()))
        colormap_colors = first_layer.colormap.colors
        contrast_limits = list(first_layer.contrast_limits)
        if invert:
            # Second component: fraction is 1 - first, so reverse the colormap
            # and the contrast limits to keep colors consistent with the
            # component line gradient in the phasor plot.
            colormap_colors = np.asarray(colormap_colors)[::-1]
            contrast_limits = [
                1.0 - contrast_limits[1],
                1.0 - contrast_limits[0],
            ]

        self.histogram_widget.update_colormap(
            colormap_colors=colormap_colors,
            contrast_limits=contrast_limits,
            gamma=first_layer.gamma,
        )

        # Ensure the range slider uses the full original data extent.
        # Without this, the default slider span (0..100 with factor=1000)
        # limits the max to 0.1 and causes max values to snap/reset.
        original_arrays = []
        for fl in fraction_layers_map.values():
            original = fl.metadata.get('fraction_data_original', fl.data)
            valid = np.asarray(original, dtype=float)
            if invert:
                valid = 1.0 - valid
            valid = valid[np.isfinite(valid)]
            if valid.size > 0:
                original_arrays.append(valid)

        if original_arrays:
            pooled = np.concatenate(original_arrays)
            data_min = float(np.min(pooled))
            data_max = float(np.max(pooled))
            if data_max <= data_min:
                data_max = data_min + 0.01

            # Keep the current active display range from the layer, but ensure
            # slider bounds span the full data extent.
            range_min = float(contrast_limits[0])
            range_max = float(contrast_limits[1])
            range_min = max(data_min, min(range_min, data_max))
            range_max = max(data_min, min(range_max, data_max))
            if range_max <= range_min:
                range_min = data_min
                range_max = data_max

            # ``colormap_contrast_limits`` drives the phasor-plot line gradient,
            # which is always expressed in first-component fraction space.
            if invert:
                self.colormap_contrast_limits = [
                    1.0 - range_max,
                    1.0 - range_min,
                ]
            else:
                self.colormap_contrast_limits = [range_min, range_max]
            self.histogram_widget.update_colormap(
                colormap_colors=colormap_colors,
                contrast_limits=[range_min, range_max],
                gamma=first_layer.gamma,
            )

            self.histogram_widget.set_range(
                range_min,
                range_max,
                slider_min=data_min,
                slider_max=data_max,
            )

        # Pool the data from every selected layer into a single merged
        # histogram so all layers contribute, rather than showing a per-layer
        # mean +/- SD that visually resembles a single layer.
        if len(fraction_layers_map) > 1:
            pooled_layers = [
                (
                    1.0 - np.asarray(fl.data, dtype=float)
                    if invert
                    else np.asarray(fl.data, dtype=float)
                ).ravel()
                for fl in fraction_layers_map.values()
            ]
            self.histogram_widget.update_data(np.concatenate(pooled_layers))
        else:
            data = first_layer.data
            if invert:
                data = 1.0 - np.asarray(data, dtype=float)
            self.histogram_widget.update_data(data)

        self.histogram_widget.show()

    def closeEvent(self, event):
        """Clean up signal connections before closing."""
        # Disconnect parent widget signal if present
        if hasattr(self, 'parent_widget') and self.parent_widget:
            with contextlib.suppress(TypeError, ValueError, AttributeError):
                self.parent_widget.harmonic_spinbox.valueChanged.disconnect(
                    self._on_harmonic_changed
                )

        event.accept()


def draw_fraction_histogram_overlay(
    ax,
    x1,
    y1,
    x2,
    y2,
    values,
    fractions_colormap,
    *,
    height=0.3,
    offset=0.0,
    alpha=0.75,
    bins=150,
    contrast_limits=None,
    gamma=1.0,
):
    """Draw the first-component fraction histogram along the component line.

    Stateless helper shared by the interactive Components tab and the batch
    export so both render identically. The histogram of ``values`` (the
    first-component fraction) uses the line joining the two components as its
    baseline and rises perpendicular to it; the area under the curve is filled
    with a seamless colormap gradient (a single ``imshow`` clipped to the curve
    polygon and transformed into the line's frame).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes hosting the phasor plot.
    x1, y1, x2, y2 : float
        Endpoints of the (possibly offset) component line: component 1 at
        ``(x1, y1)``, component 2 at ``(x2, y2)``.
    values : array-like
        First-component fraction values (any shape); flattened and filtered.
    fractions_colormap : array-like or None
        Nx4 RGBA color ramp for the gradient (falls back to ``jet``).
    height : float
        Peak histogram height as a fraction of the line length.
    offset : float
        Signed perpendicular offset of the histogram relative to the line.
        Positive keeps it on the "up" side; negative flips it to the other side.
    alpha : float
        Opacity of the gradient fill.
    bins : int
        Number of histogram bins.
    contrast_limits : sequence of float, optional
        ``(vmin, vmax)`` in first-component fraction space used to normalize the
        gradient. When given, the gradient tracks the fraction layer's contrast
        limits / range slider; otherwise it falls back to the data extent.

    Returns
    -------
    list of matplotlib artists, or ``None`` if nothing was drawn.
    """
    values = np.asarray(values, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None

    counts, bin_edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    if counts.max() == 0:
        return None
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Smooth the curve for a clean profile, matching the histogram widget.
    counts = counts.astype(float)
    with contextlib.suppress(Exception):
        import scipy.ndimage

        counts = scipy.ndimage.gaussian_filter1d(counts, sigma=2)
    peak = counts.max()
    if peak <= 0:
        return None

    # Geometry of the (possibly offset) component line.
    dx = x2 - x1
    dy = y2 - y1
    length = np.hypot(dx, dy)
    if length == 0:
        return None
    ux, uy = dx / length, dy / length
    # Base normal, pointing "up" on screen for a consistent orientation.
    nx, ny = -uy, ux
    if ny < 0:
        nx, ny = -nx, -ny

    # Signed histogram offset relative to the line: positive keeps the
    # histogram on the "up" side, negative flips it to the other side. The
    # magnitude displaces the baseline from the line by that distance.
    rise_sign = -1.0 if offset < 0 else 1.0
    rdx, rdy = rise_sign * nx, rise_sign * ny
    origin_x = x1 + offset * nx
    origin_y = y1 + offset * ny

    max_height = height * length
    heights = counts / peak * max_height
    v_max = float(np.max(heights))
    if v_max <= 0:
        return None

    # Colormap consistent with the component line.
    if fractions_colormap is not None and len(fractions_colormap) > 0:
        if len(fractions_colormap) <= 32:
            cmap = LinearSegmentedColormap.from_list(
                "fractions_interp", fractions_colormap, N=256
            )
        else:
            cmap = ListedColormap(fractions_colormap)
    else:
        cmap = plt.cm.jet

    # Normalize the gradient to the fraction layer's contrast limits (or range
    # slider) when provided, so the overlay tracks those controls. Fall back to
    # the full data extent otherwise.
    if contrast_limits is not None:
        vmin = float(contrast_limits[0])
        vmax = float(contrast_limits[1])
    else:
        vmin = float(np.min(values))
        vmax = float(np.max(values))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    gamma = gamma or 1.0
    if gamma != 1.0:
        norm = PowerNorm(gamma, vmin=vmin, vmax=vmax)
    else:
        norm = plt.Normalize(vmin=vmin, vmax=vmax)

    alpha = min(1.0, max(0.0, alpha))

    # Local (u, v) frame: u runs along the line (u=0 at component 1,
    # u=length at component 2), v rises perpendicular. A point (u, v) maps to
    # data coordinates via ``origin + u*along + v*rise``.
    local_to_data = Affine2D(
        matrix=np.array(
            [
                [ux, rdx, origin_x],
                [uy, rdy, origin_y],
                [0.0, 0.0, 1.0],
            ]
        )
    )
    transform = local_to_data + ax.transData

    # Gradient image varying along u (fraction 1 at component 1 -> 0 at
    # component 2). Preserve the plot's view limits, which ``imshow`` would
    # otherwise try to rescale.
    prev_xlim = ax.get_xlim()
    prev_ylim = ax.get_ylim()
    gradient = np.linspace(1.0, 0.0, 256).reshape(1, -1)
    im = ax.imshow(
        gradient,
        aspect="auto",
        extent=[0.0, length, 0.0, v_max],
        origin="lower",
        cmap=cmap,
        norm=norm,
        alpha=alpha,
        interpolation="bilinear",
        zorder=11,
    )
    im.set_transform(transform)
    ax.set_xlim(prev_xlim)
    ax.set_ylim(prev_ylim)

    # Clip the gradient to the area under the (smoothed) histogram curve.
    u_curve = (1.0 - bin_centers) * length
    clip_verts = np.column_stack(
        [
            np.concatenate([u_curve, u_curve[::-1]]),
            np.concatenate([heights, np.zeros_like(heights)]),
        ]
    )
    clip_poly = MplPolygon(clip_verts, closed=True, transform=transform)
    im.set_clip_path(clip_poly)

    return [im]


def draw_components_overlay(
    ax,
    reals,
    imags,
    names=None,
    colors=None,
    analysis_type="Linear Projection",
    settings=None,
):
    """Stateless function to draw components and their connecting lines or polygons on a matplotlib axes."""
    import contextlib

    import numpy as np
    from matplotlib.collections import LineCollection
    from matplotlib.colors import (
        LinearSegmentedColormap,
        ListedColormap,
        PowerNorm,
    )
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    if settings is None:
        settings = {}

    line_offset = settings.get("line_offset", 0.0)
    line_width = settings.get("line_width", 2)
    line_alpha = settings.get("line_alpha", 1.0)
    show_colormap_line = settings.get("show_colormap_line", False)
    fractions_colormap = settings.get("fractions_colormap")
    colormap_contrast_limits = settings.get("colormap_contrast_limits", (0, 1))
    colormap_gamma = settings.get("colormap_gamma", 1.0) or 1.0
    label_fontsize = settings.get("label_fontsize", 10)
    label_fontweight = settings.get("label_fontweight", "normal")
    label_fontstyle = settings.get("label_fontstyle", "normal")
    # When set, labels use this color instead of the per-component dot color.
    label_color = settings.get("label_color")
    show_dots = settings.get("show_dots", True)
    show_labels = settings.get("show_labels", True)
    default_component_color = settings.get(
        "default_component_color", "dimgray"
    )
    component_colors = settings.get(
        "component_colors",
        [
            "#e6194B",
            "#3cb44b",
            "#ffe119",
            "#4363d8",
            "#f58231",
            "#911eb4",
            "#42d4f4",
            "#f032e6",
            "#bfef45",
            "#fabed4",
            "#469990",
            "#dcbeff",
            "#9A6324",
            "#fffac8",
            "#800000",
            "#aaffc3",
            "#808000",
            "#ffd8b1",
            "#000075",
            "#a9a9a9",
        ],
    )

    if names is None:
        names = [f"Component {i+1}" for i in range(len(reals))]
    if colors is None:
        colors = [
            component_colors[i % len(component_colors)]
            for i in range(len(reals))
        ]

    num_components = len(reals)

    # 1. Draw lines or polygons first (so they are under the dots)
    if num_components == 2:
        ox1, oy1 = reals[0], imags[0]
        ox2, oy2 = reals[1], imags[1]

        if line_offset != 0.0:
            vx = ox2 - ox1
            vy = oy2 - oy1
            length = np.hypot(vx, vy)
            if length > 0:
                nx = -vy / length
                ny = vx / length
                ox1 += nx * line_offset
                oy1 += ny * line_offset
                ox2 += nx * line_offset
                oy2 += ny * line_offset

        use_colormap = (
            analysis_type == "Linear Projection"
            and show_colormap_line
            and fractions_colormap is not None
        )
        if use_colormap:
            dx = ox2 - ox1
            dy = oy2 - oy1
            length = np.hypot(dx, dy)
            if length > 0:
                t_values = np.linspace(0, 1, 500)
                trajectory_real = ox1 + t_values * dx
                trajectory_imag = oy1 + t_values * dy

                density_factor = 2
                num_segments = min(
                    len(trajectory_real) * density_factor,
                    len(trajectory_real) - 1,
                )

                if len(fractions_colormap) <= 32:
                    colormap = LinearSegmentedColormap.from_list(
                        "fractions_interp", fractions_colormap, N=256
                    )
                else:
                    colormap = ListedColormap(fractions_colormap)

                vmin, vmax = (
                    colormap_contrast_limits
                    if colormap_contrast_limits
                    else (0, 1)
                )
                segments = []
                segment_colors = []

                for i in range(num_segments):
                    start_idx = int(
                        i * (len(trajectory_real) - 1) / num_segments
                    )
                    end_idx = int(
                        (i + 1) * (len(trajectory_real) - 1) / num_segments
                    )
                    end_idx = min(end_idx, len(trajectory_real) - 1)
                    if i > 0:
                        start_idx = max(0, start_idx - 1)

                    segments.append(
                        [
                            (
                                trajectory_real[start_idx],
                                trajectory_imag[start_idx],
                            ),
                            (
                                trajectory_real[end_idx],
                                trajectory_imag[end_idx],
                            ),
                        ]
                    )

                    t = (
                        start_idx / (len(trajectory_real) - 1)
                        if len(trajectory_real) > 1
                        else 0
                    )
                    segment_colors.append(1.0 - t)

                lc = LineCollection(
                    segments, cmap=colormap, linewidths=line_width, zorder=10
                )
                lc.set_array(np.array(segment_colors))
                if colormap_gamma != 1.0 and vmax > vmin:
                    lc.set_norm(
                        PowerNorm(colormap_gamma, vmin=vmin, vmax=vmax)
                    )
                else:
                    lc.set_clim(vmin, vmax)
                lc.set_alpha(line_alpha)
                with contextlib.suppress(Exception):
                    lc.set_capstyle('butt')
                ax.add_collection(lc)
        else:
            line = ax.plot(
                [ox1, ox2],
                [oy1, oy2],
                color=default_component_color,
                linewidth=line_width,
                alpha=line_alpha,
                zorder=10,
            )[0]
            with contextlib.suppress(Exception):
                line.set_solid_capstyle('butt')

        # Optional first-component fraction histogram overlaid on the line,
        # matching the interactive Components tab.
        fraction_data = settings.get("fraction_data")
        if (
            analysis_type == "Linear Projection"
            and settings.get("show_fraction_histogram")
            and fraction_data is not None
            and fractions_colormap is not None
        ):
            draw_fraction_histogram_overlay(
                ax,
                ox1,
                oy1,
                ox2,
                oy2,
                fraction_data,
                fractions_colormap,
                height=settings.get("histogram_overlay_height", 0.3),
                offset=settings.get("histogram_offset", 0.0),
                alpha=settings.get("histogram_alpha", 0.75),
                contrast_limits=colormap_contrast_limits,
                gamma=colormap_gamma,
            )

    elif num_components >= 3:
        vertices = [(reals[i], imags[i]) for i in range(num_components)]
        vertices.append((reals[0], imags[0]))
        codes = (
            [Path.MOVETO]
            + [Path.LINETO] * (num_components - 1)
            + [Path.CLOSEPOLY]
        )
        path = Path(vertices, codes)
        patch = PathPatch(
            path,
            facecolor="lightgray",
            edgecolor=default_component_color,
            alpha=0.3 * line_alpha,
            linewidth=line_width,
            zorder=9,
        )
        ax.add_patch(patch)

    # 2. Draw dots and labels on top
    for i in range(num_components):
        real, imag = reals[i], imags[i]
        color = colors[i] if i < len(colors) else default_component_color
        name = names[i] if i < len(names) else f"C{i+1}"

        if show_dots:
            ax.plot(
                [real],
                [imag],
                marker='o',
                markersize=8,
                color=color,
                zorder=11,
            )
        if show_labels and name:
            ax.text(
                real,
                imag,
                f" {name}",
                verticalalignment='bottom',
                horizontalalignment='left',
                color=label_color or color,
                fontsize=label_fontsize,
                fontweight=label_fontweight,
                fontstyle=label_fontstyle,
                zorder=12,
            )
