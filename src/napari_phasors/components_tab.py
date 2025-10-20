from dataclasses import dataclass
from typing import TYPE_CHECKING

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from napari.experimental import link_layers
from napari.layers import Image
from napari.utils.colormaps import Colormap
from napari.utils.notifications import show_error, show_info, show_warning
from phasorpy.component import phasor_component_fit, phasor_component_fraction
from phasorpy.lifetime import phasor_from_lifetime
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
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
    text_offset: tuple[float, float] = (0.02, 0.02)
    label: str = "Component"


class ComponentsWidget(QWidget):
    """Widget to perform component analysis on phasor coordinates."""

    def __init__(self, viewer: "napari.viewer.Viewer", parent=None):
        super().__init__()
        self.viewer = viewer
        self.parent_widget = parent

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

        # Harmonic-specific component storage for multi-component analysis
        self.component_locations = {}  # {harmonic: [comp_data, ...]}
        self.current_harmonic = 1

        # Line settings
        self.show_colormap_line = True
        self.show_component_dots = True
        self.line_offset = 0.0
        self.line_width = 3.0
        self.line_alpha = 1
        self.default_component_color = 'dimgray'

        # Flag to prevent clearing lifetime when updating from lifetime
        self._updating_from_lifetime = False

        # Flag to track if analysis was attempted
        self._analysis_attempted = False

        # Dialog / event flags
        self.plot_dialog = None
        self.style_dialog = None
        self.drag_events_connected = False

        # Drag state
        self.dragging_component_idx = None
        self.dragging_label_idx = None

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

        # Scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        root_layout.addWidget(scroll_area)

        # Content widget inside scroll area
        content_widget = QWidget()
        scroll_area.setWidget(content_widget)

        # Previous main layout becomes content_layout
        layout = QVBoxLayout()
        content_widget.setLayout(layout)

        # Analysis type selection
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
        layout.addLayout(analysis_layout)

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
        layout.addWidget(self.components_info_label)

        # Components section
        self.components_layout = QVBoxLayout()
        layout.addLayout(self.components_layout)

        # Initialize with 2 components
        for i in range(2):
            self._add_component_ui(i)

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
        layout.addLayout(comp_management_layout)

        # Plot and label settings section
        settings_label = QLabel("Display Settings:")
        settings_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        layout.addWidget(settings_label)

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
        layout.addLayout(buttons_row)

        # Calculate button
        self.calculate_button = QPushButton("Run Analysis")
        self.calculate_button.clicked.connect(self._run_analysis)
        self.calculate_button.setToolTip(
            "Run the selected analysis type on the defined components."
        )
        layout.addWidget(self.calculate_button)

        layout.addStretch()
        self.setLayout(root_layout)

        # Update component visibility and button states
        self._update_component_visibility()
        self._update_button_states()

    def _add_component_ui(self, idx):
        """Add UI elements for a component."""
        # Single component layout with all elements in one row
        comp_layout = QHBoxLayout()

        # Component name
        name_edit = QLineEdit()
        name_edit.setPlaceholderText("Component name (optional)")
        name_edit.setMaximumWidth(150)
        name_edit.setToolTip("Enter a name for this component (optional).")
        comp_layout.addWidget(name_edit)

        # Select button
        select_button = QPushButton("Select location")
        select_button.setMaximumWidth(100)
        select_button.setToolTip(
            "Click here and then click on the phasor plot to select the component location."
        )
        comp_layout.addWidget(select_button)

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
        s_edit.editingFinished.connect(
            lambda: self._on_component_coords_changed(idx)
        )
        s_edit.textChanged.connect(
            lambda: self._update_component_input_styling(idx)
        )
        lifetime_edit.editingFinished.connect(
            lambda: self._update_component_from_lifetime(idx)
        )
        select_button.clicked.connect(lambda: self._select_component(idx))

        # Store UI elements for visibility control
        comp.ui_elements = {
            'comp_layout': comp_layout,
            'lifetime_label': lifetime_label,
        }

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

        if len(self.components) == 2:
            self._update_analysis_options()

            index = self.analysis_type_combo.findText("Linear Projection")
            if index >= 0:
                self.analysis_type_combo.setCurrentIndex(index)

        if self.component_line is not None:
            try:
                self.component_line.remove()
            except (ValueError, AttributeError):
                pass
            self.component_line = None

        if self.component_polygon is not None:
            try:
                self.component_polygon.remove()
            except (ValueError, AttributeError):
                pass
            self.component_polygon = None

        self._update_component_visibility()
        self._update_analysis_options()
        self._update_button_states()

        # Update styling after removing component (only if analysis was attempted)
        if self._analysis_attempted:
            self._update_all_component_styling()

        self.draw_line_between_components()

        if self.parent_widget is not None and hasattr(
            self.parent_widget, '_get_frequency_from_layer'
        ):
            self._update_lifetime_inputs_visibility()

        if self.parent_widget is not None:
            self.parent_widget.canvas_widget.canvas.draw_idle()

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
        except Exception:
            return 3

    def _update_button_states(self):
        """Update the enabled/disabled state of add/remove component buttons."""
        total_count = len([c for c in self.components if c is not None])
        max_components = self._get_max_components()

        # Add button: enabled if less than max components
        self.add_component_btn.setEnabled(total_count < max_components)

        # Remove button: enabled if more than 2 components
        self.remove_component_btn.setEnabled(total_count > 2)

    def _clear_components(self):
        """Clear all components."""
        for comp in self.components:
            if comp is not None:
                if comp.dot is not None:
                    comp.dot.remove()
                if comp.text is not None:
                    comp.text.remove()
                comp.g_edit.clear()
                comp.s_edit.clear()
                comp.name_edit.clear()
                if comp.lifetime_edit is not None:
                    comp.lifetime_edit.clear()

        if self.component_line is not None:
            try:
                self.component_line.remove()
            except (ValueError, AttributeError):
                pass
            self.component_line = None

        if self.component_polygon is not None:
            try:
                self.component_polygon.remove()
            except (ValueError, AttributeError):
                pass
            self.component_polygon = None

        for comp in self.components:
            if comp is not None:
                comp.dot = None
                comp.text = None

        if self.parent_widget is not None:
            self.parent_widget.canvas_widget.canvas.draw_idle()

    def _open_plot_settings_dialog(self):
        """Open dialog to edit plot settings."""
        if self.plot_dialog is not None and self.plot_dialog.isVisible():
            self.plot_dialog.raise_()
            self.plot_dialog.activateWindow()
            return

        self.plot_dialog = QDialog(self)
        self.plot_dialog.setWindowTitle("Component Line Settings")
        vbox = QVBoxLayout(self.plot_dialog)

        # Row 1: checkboxes
        row1 = QHBoxLayout()
        self.colormap_line_checkbox = QCheckBox("Overlay Colormap")
        self.colormap_line_checkbox.setChecked(self.show_colormap_line)
        self.colormap_line_checkbox.stateChanged.connect(
            self._on_plot_setting_changed
        )
        row1.addWidget(self.colormap_line_checkbox)

        self.show_dots_checkbox = QCheckBox("Show component positions")
        self.show_dots_checkbox.setChecked(self.show_component_dots)
        self.show_dots_checkbox.stateChanged.connect(
            self._on_plot_setting_changed
        )
        row1.addWidget(self.show_dots_checkbox)
        row1.addStretch()
        vbox.addLayout(row1)

        # Row 2: line offset
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Line offset:"))
        self.line_offset_slider = QSlider(Qt.Horizontal)
        self.line_offset_slider.setRange(-500, 500)
        self.line_offset_slider.setValue(int(self.line_offset * 1000))
        self.line_offset_slider.valueChanged.connect(
            self._on_line_offset_changed
        )
        row2.addWidget(self.line_offset_slider)
        self.line_offset_value_label = QLabel(f"{self.line_offset:.3f}")
        row2.addWidget(self.line_offset_value_label)
        vbox.addLayout(row2)

        # Row 3: width & alpha
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Line width:"))
        self.line_width_spin = QDoubleSpinBox()
        self.line_width_spin.setRange(0.5, 20.0)
        self.line_width_spin.setSingleStep(0.5)
        self.line_width_spin.setValue(self.line_width)
        self.line_width_spin.valueChanged.connect(self._on_line_width_changed)
        row3.addWidget(self.line_width_spin)

        row3.addWidget(QLabel("Alpha:"))
        self.line_alpha_slider = QSlider(Qt.Horizontal)
        self.line_alpha_slider.setRange(0, 100)
        self.line_alpha_slider.setValue(int(self.line_alpha * 100))
        self.line_alpha_slider.valueChanged.connect(
            self._on_line_alpha_changed
        )
        row3.addWidget(self.line_alpha_slider)
        self.line_alpha_value_label = QLabel(f"{self.line_alpha:.2f}")
        row3.addWidget(self.line_alpha_value_label)

        row3.addStretch()
        vbox.addLayout(row3)

        # Row 4: Color selection
        row4 = QHBoxLayout()
        row4.addWidget(QLabel("Default color:"))
        self.color_button = QPushButton()
        self.color_button.setMaximumWidth(80)
        self.color_button.setStyleSheet(
            f"background-color: {self.default_component_color}; border: 1px solid black;"
        )
        self.color_button.clicked.connect(self._on_color_button_clicked)
        row4.addWidget(self.color_button)

        row4.addStretch()
        vbox.addLayout(row4)

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

    def _on_harmonic_changed(self, new_harmonic):
        """Handle harmonic changes - store current components and restore for new harmonic."""
        old_harmonic = self.current_harmonic
        self._store_current_components(old_harmonic)
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
        """Calculate minimum number of harmonics required for given number of components."""
        if num_components <= 3:
            return 1
        else:
            # For more than 3 components: num_components <= 2 * num_harmonics + 1
            # Solving for num_harmonics: num_harmonics >= (num_components - 1) / 2
            return max(2, int(np.ceil((num_components - 1) / 2)))

    def _store_current_components(self, harmonic):
        """Store current component locations for the given harmonic."""
        if harmonic not in self.component_locations:
            self.component_locations[harmonic] = []

        self.component_locations[harmonic] = []

        for comp in self.components:
            if comp is not None and comp.dot is not None:
                x_data, y_data = comp.dot.get_data()
                name = comp.name_edit.text().strip()

                if not name or name.startswith("Component "):
                    name = ""

                comp_data = {
                    'g': x_data[0],
                    's': y_data[0],
                    'name': name,
                    'idx': comp.idx,
                }

                while len(self.component_locations[harmonic]) <= comp.idx:
                    self.component_locations[harmonic].append(None)

                self.component_locations[harmonic][comp.idx] = comp_data
            else:
                while (
                    len(self.component_locations[harmonic]) <= comp.idx
                    if comp is not None
                    else 0
                ):
                    self.component_locations[harmonic].append(None)

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
            try:
                self.component_polygon.remove()
            except (ValueError, AttributeError):
                pass
            self.component_polygon = None

        if self.component_line is not None:
            try:
                self.component_line.remove()
            except (ValueError, AttributeError):
                pass
            self.component_line = None

        if self.parent_widget is not None:
            self.parent_widget.canvas_widget.canvas.draw_idle()

    def _restore_components_for_harmonic(self, harmonic):
        """Restore component locations for the given harmonic."""
        if harmonic not in self.component_locations:
            return

        stored_components = self.component_locations[harmonic]

        for i, comp_data in enumerate(stored_components):
            if (
                comp_data is not None
                and i < len(self.components)
                and self.components[i] is not None
            ):
                comp = self.components[i]

                comp.g_edit.setText(f"{comp_data['g']:.6f}")
                comp.s_edit.setText(f"{comp_data['s']:.6f}")

                stored_name = comp_data.get('name', '')
                current_name = comp.name_edit.text().strip()

                if stored_name and (
                    not current_name or current_name.startswith("Component ")
                ):
                    comp.name_edit.setText(stored_name)

                self._create_component_at_coordinates(
                    i, comp_data['g'], comp_data['s']
                )

        self.draw_line_between_components()
        self._update_component_colors()
        if self.parent_widget is not None:
            self.parent_widget.canvas_widget.canvas.draw_idle()

    def _run_analysis(self):
        """Run the selected analysis."""
        self._analysis_attempted = True

        self._update_all_component_styling()

        if self.analysis_type == "Linear Projection":
            self._run_linear_projection()
        else:
            self._run_component_fit()

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

        self.show_colormap_line = default_show_colormap_line
        self.show_component_dots = default_show_component_dots
        self.line_offset = default_line_offset
        self.line_width = default_line_width
        self.line_alpha = default_line_alpha
        self.default_component_color = default_component_color

        self.colormap_line_checkbox.setChecked(default_show_colormap_line)
        self.show_dots_checkbox.setChecked(default_show_component_dots)

        self.line_offset_slider.setValue(int(default_line_offset * 1000))
        self.line_offset_value_label.setText(f"{default_line_offset:.3f}")

        self.line_width_spin.setValue(default_line_width)

        self.line_alpha_slider.setValue(int(default_line_alpha * 100))
        self.line_alpha_value_label.setText(f"{default_line_alpha:.2f}")

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

    def _toggle_plot_section(self, checked):
        """Toggle visibility of the plot section."""
        self.plot_section.setVisible(checked)

    def _on_plot_setting_changed(self):
        """Handle changes to plot settings from dialog."""
        self.show_colormap_line = self.colormap_line_checkbox.isChecked()
        self.show_component_dots = self.show_dots_checkbox.isChecked()

        active_components = [
            c for c in self.components if c is not None and c.dot is not None
        ]

        if (
            len(active_components) == 2
            and self.show_colormap_line
            and self.fractions_colormap is not None
        ):
            self._update_component_colors()
        elif len(active_components) > 2:
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
        """Handle changes to line offset from slider."""
        self.line_offset = value / 1000.0
        self.line_offset_value_label.setText(f"{self.line_offset:.3f}")
        self.draw_line_between_components()
        if self.parent_widget is not None:
            self.parent_widget.canvas_widget.canvas.draw_idle()

    def _on_line_width_changed(self, value):
        """Handle changes to line width from spinbox."""
        self.line_width = float(value)

        if isinstance(self.component_line, LineCollection):
            try:
                self.component_line.set_linewidths([self.line_width])
            except Exception:
                pass

            if self.parent_widget is not None:
                self.parent_widget.canvas_widget.canvas.draw_idle()
        else:
            self.draw_line_between_components()
            if self.parent_widget is not None:
                self.parent_widget.canvas_widget.canvas.draw_idle()

    def _on_line_alpha_changed(self, value):
        """Handle changes to line alpha from slider."""
        self.line_alpha = value / 100.0
        self.line_alpha_value_label.setText(f"{self.line_alpha:.2f}")

        if self.component_line is not None:
            if hasattr(self.component_line, 'set_alpha'):
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
            self._apply_styles_to_labels()

    def _on_label_style_changed(self):
        """Handle changes to label style settings."""
        self.label_fontsize = self.fontsize_spin.value()
        self.label_bold = self.bold_checkbox.isChecked()
        self.label_italic = self.italic_checkbox.isChecked()
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

        for comp in self.components:
            if (
                comp is not None
                and hasattr(comp, 'ui_elements')
                and 'lifetime_label' in comp.ui_elements
            ):
                comp.ui_elements['lifetime_label'].setVisible(has_freq)
                if comp.lifetime_edit is not None:
                    comp.lifetime_edit.setVisible(has_freq)

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

        # Only highlight if analysis was attempted
        if not self._analysis_attempted:
            return

        comp = self.components[idx]

        # Check if both G and S have values
        has_g_value = comp.g_edit.text().strip() != ""
        has_s_value = comp.s_edit.text().strip() != ""

        if has_g_value and has_s_value:
            # Reset to default styling
            comp.g_edit.setStyleSheet("")
            comp.s_edit.setStyleSheet("")
        else:
            # Check if we need to highlight (component locations required)
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

    def _compute_phasor_from_lifetime(self, lifetime_text):
        """Compute (G,S) from lifetime string; return tuple or (None,None)."""
        try:
            lifetime = float(lifetime_text)
        except (TypeError, ValueError):
            return None, None
        freq = self.parent_widget._get_frequency_from_layer()
        if freq is None:
            return None, None

        harmonic = getattr(self.parent_widget, "harmonic", 1)
        re, im = phasor_from_lifetime(freq * harmonic, lifetime)
        if np.ndim(re) > 0:
            re = float(np.array(re).ravel()[0])
        if np.ndim(im) > 0:
            im = float(np.array(im).ravel()[0])
        return re, im

    def _update_component_from_lifetime(self, idx: int):
        """Update component G/S coordinates based on lifetime input."""
        comp = self.components[idx]
        txt = comp.lifetime_edit.text().strip()
        if not txt:
            return
        G, S = self._compute_phasor_from_lifetime(txt)
        if G is None:
            return

        self._updating_from_lifetime = True
        comp.g_edit.setText(f"{G:.6f}")
        comp.s_edit.setText(f"{S:.6f}")
        self._on_component_coords_changed(idx)
        self._updating_from_lifetime = False

    def _on_component_coords_changed(self, idx: int):
        """Handle changes to component G/S coordinates from text inputs."""
        comp = self.components[idx]
        try:
            x = float(comp.g_edit.text())
            y = float(comp.s_edit.text())
        except ValueError:
            return

        if comp.lifetime_edit is not None and not self._updating_from_lifetime:
            comp.lifetime_edit.clear()

        if comp.dot is not None:
            comp.dot.set_data([x], [y])
            if comp.text is not None:
                ox, oy = comp.text_offset
                comp.text.set_position((x + ox, y + oy))
            self.draw_line_between_components()
        else:
            self._create_component_at_coordinates(idx, x, y)

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
            except Exception:
                colors.append(
                    self.component_colors[i % len(self.component_colors)]
                )

        return colors

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

        except Exception:
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
        """Get component coordinates for a specific harmonic."""
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
        elif harmonic in self.component_locations:
            stored_components = self.component_locations[harmonic]
            for comp_data in stored_components:
                if comp_data is not None:
                    component_g.append(comp_data['g'])
                    component_s.append(comp_data['s'])
                    stored_name = comp_data.get('name', '')

                    if stored_name:
                        component_names.append(stored_name)
                    else:
                        component_names.append(
                            f"Component {comp_data['idx']+1}"
                        )

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

    def _on_component_name_changed(self, idx: int):
        """Update the label text for a component when its name is changed."""
        comp = self.components[idx]
        if comp.dot is None:
            return
        name = comp.name_edit.text().strip()
        prev_pos = None
        if comp.text is not None:
            prev_pos = comp.text.get_position()
            try:
                comp.text.remove()
            except (ValueError, AttributeError):
                pass
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

    def _select_component(self, idx: int):
        """Activate selection mode for a component to pick its location."""
        if self.parent_widget is None:
            return

        self.parent_widget.canvas_widget._on_escape(None)

        comp = self.components[idx]

        if comp.dot is not None:
            comp.dot.set_visible(False)
        if comp.text is not None:
            comp.text.set_visible(False)
        if self.component_line is not None:
            self.component_line.set_visible(False)
        self._redraw(force=True)

        original_text = comp.select_button.text()
        comp.select_button.setText("Click on plot...")
        comp.select_button.setEnabled(False)

        temp_cid = self.parent_widget.canvas_widget.canvas.mpl_connect(
            'button_press_event',
            lambda event: self._handle_component_selection(
                event, idx, temp_cid, original_text
            ),
        )

    def _handle_component_selection(self, event, idx, temp_cid, original_text):
        """Handle the selection of a component by clicking on the plot."""
        if not event.inaxes:
            return

        comp = self.components[idx]
        x, y = event.xdata, event.ydata
        comp.g_edit.setText(f"{x:.6f}")
        comp.s_edit.setText(f"{y:.6f}")

        if comp.lifetime_edit is not None:
            comp.lifetime_edit.clear()

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

        self.parent_widget.canvas_widget.canvas.mpl_disconnect(temp_cid)
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
            if (
                hasattr(self, 'fractions_colormap')
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
                            if hasattr(cmap, '__call__'):

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
                    except Exception:
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
            try:
                self.component_line.remove()
            except (ValueError, AttributeError):
                pass
            self.component_line = None

        if self.component_polygon is not None:
            try:
                self.component_polygon.remove()
            except (ValueError, AttributeError):
                pass
            self.component_polygon = None

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
                self.show_colormap_line
                and self.comp1_fractions_layer is not None
                and self.fractions_colormap is not None
            )

            if use_colormap:
                self._draw_colormap_line(ax, ox1, oy1, ox2, oy2)
                self._update_component_colors()
                if hasattr(self.component_line, "set_capstyle"):
                    try:
                        self.component_line.set_capstyle('butt')
                    except Exception:
                        pass

                if hasattr(self.component_line, 'set_zorder'):
                    try:
                        self.component_line.set_zorder(10)
                    except Exception:
                        pass
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
                    try:
                        self.component_line.set_solid_capstyle('butt')
                    except Exception:
                        pass

                self._update_component_colors()

            self.parent_widget.canvas_widget.canvas.draw_idle()

            components_tab_is_active = (
                self.parent_widget is not None
                and getattr(self.parent_widget, "tab_widget", None) is not None
                and self.parent_widget.tab_widget.currentWidget()
                is self.parent_widget.components_tab
            )
            self.set_artists_visible(components_tab_is_active)

        except Exception as e:
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
            colormap = plt.cm.PiYG

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
        lc.set_clim(vmin, vmax)
        lc.set_alpha(self.line_alpha)

        if hasattr(lc, "set_capstyle"):
            try:
                lc.set_capstyle('butt')
            except Exception:
                pass
        self.component_line = ax.add_collection(lc)

    def _update_polygon(self):
        """Update polygon for multi-component visualization."""
        if self.component_polygon is not None:
            try:
                self.component_polygon.remove()
            except (ValueError, AttributeError):
                pass
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
        """Handle changes to the colormap of the fractions layer."""
        if (
            self.comp1_fractions_layer is not None
            and self.component_line is not None
        ):
            layer = event.source
            self.fractions_colormap = layer.colormap.colors
            self.colormap_contrast_limits = layer.contrast_limits

            self.draw_line_between_components()

    def _on_contrast_limits_changed(self, event):
        """Handle changes to the contrast limits of the fractions layer."""
        if (
            self.comp1_fractions_layer is not None
            and self.component_line is not None
        ):
            layer = event.source
            self.colormap_contrast_limits = layer.contrast_limits

            self.draw_line_between_components()

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
                    try:
                        self.parent_widget.canvas_widget.toolbar.release_zoom(
                            event
                        )
                    except Exception:
                        pass
                if self.parent_widget.canvas_widget.toolbar.mode == 'pan/zoom':
                    try:
                        self.parent_widget.canvas_widget.toolbar.release_pan(
                            event
                        )
                    except Exception:
                        pass
                self.parent_widget.canvas_widget._on_escape(None)
                self.dragging_label_idx = comp.idx
                return
        for comp in self.components:
            if comp.dot is not None and comp.dot.contains(event)[0]:
                if (
                    self.parent_widget.canvas_widget.toolbar.mode
                    == 'zoom rect'
                ):
                    try:
                        self.parent_widget.canvas_widget.toolbar.release_zoom(
                            event
                        )
                    except Exception:
                        pass
                if self.parent_widget.canvas_widget.toolbar.mode == 'pan/zoom':
                    try:
                        self.parent_widget.canvas_widget.toolbar.release_pan(
                            event
                        )
                    except Exception:
                        pass
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
        comp.g_edit.setText(f"{x:.6f}")
        comp.s_edit.setText(f"{y:.6f}")

        if comp.lifetime_edit is not None:
            comp.lifetime_edit.clear()

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
        """Get list of harmonics that have component data."""
        harmonics = set()

        for harmonic, stored_components in self.component_locations.items():
            if stored_components and any(
                comp is not None for comp in stored_components
            ):
                harmonics.add(harmonic)

        current_harmonic = getattr(self.parent_widget, 'harmonic', 1)
        active_components = [
            c for c in self.components if c is not None and c.dot is not None
        ]

        if len(active_components) > 0:
            harmonics.add(current_harmonic)

        return sorted(list(harmonics))

    def _get_available_harmonics(self):
        """Get available harmonics from the phasor data."""
        if self.parent_widget._labels_layer_with_phasor_features is None:
            return []

        phasor_data = (
            self.parent_widget._labels_layer_with_phasor_features.features
        )
        return sorted(phasor_data['harmonic'].unique())

    def _get_next_harmonic(self, current_harmonic, available_harmonics):
        """Get the next available harmonic after the current one."""
        for harmonic in available_harmonics:
            if harmonic > current_harmonic:
                return harmonic
        return None

    def _sync_colormaps(self, event):
        """Synchronize colormaps between comp1 and comp2 layers with inversion."""
        if (
            self.comp1_fractions_layer is None
            or self.comp2_fractions_layer is None
        ):
            return

        try:
            if event.source == self.comp1_fractions_layer:
                current_colormap = self.comp1_fractions_layer.colormap

                if (
                    hasattr(current_colormap, 'colors')
                    and current_colormap.colors is not None
                ):
                    inverted_colors = current_colormap.colors[::-1]

                    self.comp2_fractions_layer.events.colormap.disconnect(
                        self._sync_colormaps
                    )

                    inverted_cmap = Colormap(
                        colors=inverted_colors, name="inverted_custom"
                    )
                    self.comp2_fractions_layer.colormap = inverted_cmap

                    self.comp2_fractions_layer.events.colormap.connect(
                        self._sync_colormaps
                    )
                else:
                    self.comp2_fractions_layer.events.colormap.disconnect(
                        self._sync_colormaps
                    )
                    self.comp2_fractions_layer.colormap = 'PiYG_r'
                    self.comp2_fractions_layer.events.colormap.connect(
                        self._sync_colormaps
                    )

            elif event.source == self.comp2_fractions_layer:
                current_colormap = self.comp2_fractions_layer.colormap

                if (
                    hasattr(current_colormap, 'colors')
                    and current_colormap.colors is not None
                ):
                    inverted_colors = current_colormap.colors[::-1]

                    self.comp1_fractions_layer.events.colormap.disconnect(
                        self._sync_colormaps
                    )

                    inverted_cmap = Colormap(
                        colors=inverted_colors, name="inverted_custom"
                    )
                    self.comp1_fractions_layer.colormap = inverted_cmap

                    self.comp1_fractions_layer.events.colormap.connect(
                        self._sync_colormaps
                    )
                else:
                    self.comp1_fractions_layer.events.colormap.disconnect(
                        self._sync_colormaps
                    )
                    self.comp1_fractions_layer.colormap = 'PiYG'
                    self.comp1_fractions_layer.events.colormap.connect(
                        self._sync_colormaps
                    )

            if hasattr(self.comp1_fractions_layer.colormap, 'colors'):
                self.fractions_colormap = (
                    self.comp1_fractions_layer.colormap.colors
                )

            self.draw_line_between_components()

        except Exception as e:
            try:
                if self.comp2_fractions_layer is not None:
                    self.comp2_fractions_layer.events.colormap.disconnect(
                        self._sync_colormaps
                    )
                if self.comp1_fractions_layer is not None:
                    self.comp1_fractions_layer.events.colormap.disconnect(
                        self._sync_colormaps
                    )

                if hasattr(event, 'source'):
                    if event.source == self.comp1_fractions_layer:
                        if self.comp2_fractions_layer is not None:
                            self.comp2_fractions_layer.colormap = 'PiYG_r'
                    else:
                        if self.comp1_fractions_layer is not None:
                            self.comp1_fractions_layer.colormap = 'PiYG'

                if self.comp2_fractions_layer is not None:
                    self.comp2_fractions_layer.events.colormap.connect(
                        self._sync_colormaps
                    )
                if self.comp1_fractions_layer is not None:
                    self.comp1_fractions_layer.events.colormap.connect(
                        self._sync_colormaps
                    )

            except Exception:
                pass

    def _create_fraction_layers(self, fractions, component_names):
        """Create fraction layers for each component."""
        for layer in self.fraction_layers:
            try:
                self.viewer.layers.remove(layer)
            except ValueError:
                pass
        self.fraction_layers.clear()

        base_name = (
            self.parent_widget.image_layer_with_phasor_features_combobox.currentText()
        )

        for i, (fraction, name) in enumerate(zip(fractions, component_names)):
            fraction_reshaped = fraction.reshape(
                self.parent_widget._labels_layer_with_phasor_features.data.shape
            )

            layer_name = f"{name} fraction: {base_name}"

            colormap = self.component_colormap_names[
                i % len(self.component_colormap_names)
            ]

            layer = self.viewer.add_image(
                fraction_reshaped,
                name=layer_name,
                scale=self.parent_widget._labels_layer_with_phasor_features.scale,
                colormap=colormap,
                contrast_limits=(0, 1),
            )

            self.fraction_layers.append(layer)
            layer.events.colormap.connect(self._on_fraction_colormap_changed)

        self._update_component_colors()

    def _on_fraction_colormap_changed(self, event):
        """Handle changes to the colormap of any fraction layer."""
        active_components = [
            c for c in self.components if c is not None and c.dot is not None
        ]

        if len(active_components) == 2 and self.show_colormap_line:
            self._update_component_colors()
        elif len(active_components) > 2:
            self._update_component_colors()

    def _run_linear_projection(self):
        """Run linear projection for 2-component analysis."""
        if self.parent_widget._labels_layer_with_phasor_features is None:
            return
        if not all(c.dot is not None for c in self.components[:2]):
            return
        c1, c2 = self.components[:2]
        component_real = (c1.dot.get_data()[0][0], c2.dot.get_data()[0][0])
        component_imag = (c1.dot.get_data()[1][0], c2.dot.get_data()[1][0])

        phasor_data = (
            self.parent_widget._labels_layer_with_phasor_features.features
        )
        harmonic_mask = phasor_data['harmonic'] == self.parent_widget.harmonic
        real = phasor_data.loc[harmonic_mask, 'G']
        imag = phasor_data.loc[harmonic_mask, 'S']

        fractions = phasor_component_fraction(
            np.array(real), np.array(imag), component_real, component_imag
        )
        fractions = fractions.reshape(
            self.parent_widget._labels_layer_with_phasor_features.data.shape
        )

        comp1_name = c1.name_edit.text().strip() or "Component 1"
        comp1_fractions_layer_name = f"{comp1_name} fractions: {self.parent_widget.image_layer_with_phasor_features_combobox.currentText()}"
        comp2_name = c2.name_edit.text().strip() or "Component 2"
        comp2_fractions_layer_name = f"{comp2_name} fractions: {self.parent_widget.image_layer_with_phasor_features_combobox.currentText()}"

        comp1_selected_fractions_layer = Image(
            fractions,
            name=comp1_fractions_layer_name,
            scale=self.parent_widget._labels_layer_with_phasor_features.scale,
            colormap='PiYG',
            contrast_limits=(0, 1),
        )
        comp2_selected_fractions_layer = Image(
            1.0 - fractions,
            name=comp2_fractions_layer_name,
            scale=self.parent_widget._labels_layer_with_phasor_features.scale,
            colormap='PiYG_r',  # Use inverted colormap
            contrast_limits=(0, 1),
        )

        if comp1_fractions_layer_name in self.viewer.layers:
            self.viewer.layers.remove(
                self.viewer.layers[comp1_fractions_layer_name]
            )
        if comp2_fractions_layer_name in self.viewer.layers:
            self.viewer.layers.remove(
                self.viewer.layers[comp2_fractions_layer_name]
            )

        self.comp1_fractions_layer = self.viewer.add_layer(
            comp1_selected_fractions_layer
        )
        self.comp2_fractions_layer = self.viewer.add_layer(
            comp2_selected_fractions_layer
        )
        self.fractions_colormap = self.comp1_fractions_layer.colormap.colors
        self.colormap_contrast_limits = (
            self.comp1_fractions_layer.contrast_limits
        )

        self.comp1_fractions_layer.events.colormap.connect(
            self._on_colormap_changed
        )
        self.comp1_fractions_layer.events.colormap.connect(
            self._sync_colormaps
        )
        self.comp2_fractions_layer.events.colormap.connect(
            self._sync_colormaps
        )
        self.comp2_fractions_layer.events.contrast_limits.connect(
            self._on_contrast_limits_changed
        )
        link_layers(
            [self.comp1_fractions_layer, self.comp2_fractions_layer],
            ('contrast_limits', 'gamma'),
        )
        self.draw_line_between_components()

    def _run_component_fit(self):
        """Run multi-component analysis using phasor_component_fit."""
        if self.parent_widget._labels_layer_with_phasor_features is None:
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

        if required_harmonics == 1:
            component_g, component_s, component_names = (
                self._get_component_coords_for_harmonic(current_harmonic)
            )
            if not component_g:
                show_warning(
                    f"No components found for harmonic {current_harmonic}"
                )
                return

            phasor_data = (
                self.parent_widget._labels_layer_with_phasor_features.features
            )
            harmonic_mask = phasor_data['harmonic'] == current_harmonic

            layer_name = (
                self.parent_widget.image_layer_with_phasor_features_combobox.currentText()
            )
            mean = self.viewer.layers[layer_name].metadata['original_mean']

            real = np.reshape(
                phasor_data.loc[harmonic_mask, "G"].values,
                mean.shape,
            )
            imag = np.reshape(
                phasor_data.loc[harmonic_mask, "S"].values,
                mean.shape,
            )

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

            # Get phasor data for all harmonics
            phasor_data = (
                self.parent_widget._labels_layer_with_phasor_features.features
            )

            layer_name = (
                self.parent_widget.image_layer_with_phasor_features_combobox.currentText()
            )
            mean = self.viewer.layers[layer_name].metadata['original_mean']

            real_list = []
            imag_list = []

            for harmonic in harmonics_with_components:
                harmonic_mask = phasor_data['harmonic'] == harmonic
                real_h = np.reshape(
                    phasor_data.loc[harmonic_mask, "G"].values,
                    mean.shape,
                )
                imag_h = np.reshape(
                    phasor_data.loc[harmonic_mask, "S"].values,
                    mean.shape,
                )
                real_list.append(real_h)
                imag_list.append(imag_h)

            real = np.stack(
                real_list, axis=0
            )  # Shape: [num_harmonics, height, width]
            imag = np.stack(
                imag_list, axis=0
            )  # Shape: [num_harmonics, height, width]

        try:
            fractions = phasor_component_fit(
                mean, real, imag, component_g, component_s
            )

            self._create_fraction_layers(fractions, component_names)

        except Exception as e:
            show_error(f"Analysis failed: {str(e)}")
