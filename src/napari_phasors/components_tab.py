from dataclasses import dataclass
from typing import TYPE_CHECKING

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from napari.experimental import link_layers
from napari.layers import Image
from napari.utils.colormaps import AVAILABLE_COLORMAPS, Colormap
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

        # Flag to prevent clearing lifetime when updating from lifetime
        self._updating_from_lifetime = False
        self._updating_settings = False  # Flag to prevent recursive updates

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
        except Exception:
            return 3

    def _update_button_states(self):
        """Update the enabled/disabled state of add/remove component buttons."""
        total_count = len([c for c in self.components if c is not None])
        max_components = self._get_max_components()

        self.add_component_btn.setEnabled(total_count < max_components)

        self.remove_component_btn.setEnabled(total_count > 2)

    def _clear_components(self):
        """Clear all component visualizations and input fields."""
        for comp in self.components:
            if comp.dot is not None:
                try:
                    comp.dot.remove()
                except (ValueError, AttributeError):
                    pass
                comp.dot = None

            if comp.text is not None:
                try:
                    comp.text.remove()
                except (ValueError, AttributeError):
                    pass
                comp.text = None

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

        self._update_components_setting_in_metadata('components', {})

        if self.parent_widget is not None:
            self.parent_widget.canvas_widget.canvas.draw_idle()

    def _get_default_components_settings(self):
        """Get default settings dictionary for components parameters."""
        return {
            'analysis_type': 'Linear Projection',
            'components': {},
            'line_settings': {
                'show_colormap_line': True,
                'show_component_dots': True,
                'line_offset': 0.0,
                'line_width': 3.0,
                'line_alpha': 1.0,
            },
            'label_settings': {
                'fontsize': 10,
                'bold': False,
                'italic': False,
                'color': 'black',
            },
        }

    def _update_components_setting_in_metadata(self, key_path, value):
        """Update a specific component setting in the current layer's metadata."""
        layer_name = (
            self.parent_widget.image_layer_with_phasor_features_combobox.currentText()
        )
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

    def _restore_and_recreate_components_from_metadata(self):
        """Restore all components settings and recreate visual elements from metadata."""

        layer_name = (
            self.parent_widget.image_layer_with_phasor_features_combobox.currentText()
        )

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

            for comp in self.components:
                if comp is not None:
                    if comp.dot is not None:
                        comp.dot.remove()
                        comp.dot = None
                    if comp.text is not None:
                        comp.text.remove()
                        comp.text = None
                    comp.name_edit.clear()
                    if comp.lifetime_edit is not None:
                        comp.lifetime_edit.clear()
                    comp.g_edit.clear()
                    comp.s_edit.clear()

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
                    max_idx = max(
                        int(k) for k in settings['components'].keys()
                    )
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

            if 'line_settings' in settings:
                line_settings = settings['line_settings']
                self.show_colormap_line = line_settings.get(
                    'show_colormap_line', True
                )
                self.show_component_dots = line_settings.get(
                    'show_component_dots', True
                )
                self.line_offset = line_settings.get('line_offset', 0.0)
                self.line_width = line_settings.get('line_width', 3.0)
                self.line_alpha = line_settings.get('line_alpha', 1.0)

            if 'label_settings' in settings:
                label_settings = settings['label_settings']
                self.label_fontsize = label_settings.get('fontsize', 10)
                self.label_bold = label_settings.get('bold', False)
                self.label_italic = label_settings.get('italic', False)
                self.label_color = label_settings.get('color', 'black')

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
                            for harmonic_key in gs_harmonics.keys():
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

        except Exception as e:
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

            try:
                fraction_layer.events.colormap.disconnect(
                    self._on_colormap_changed
                )
            except Exception:
                pass

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

            finally:
                fraction_layer.events.colormap.connect(
                    self._on_colormap_changed
                )

        if (
            len(settings.get('components', {})) == 2
            and self.analysis_type == "Linear Projection"
        ):
            if self.comp1_fractions_layer is not None:
                self.fractions_colormap = (
                    self.comp1_fractions_layer.colormap.colors
                )
                self.colormap_contrast_limits = (
                    self.comp1_fractions_layer.contrast_limits
                )

        self._update_component_colors()
        self.draw_line_between_components()

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
        """Restore component locations for the given harmonic from metadata."""
        if not self.current_image_layer_name:
            return

        layer = self.viewer.layers[self.current_image_layer_name]
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

    def _apply_saved_colormap_settings(self):
        """Apply saved colormap settings to fraction layers if they exist."""
        if (self.comp1_fractions_layer is not None and 
            hasattr(self, '_saved_colormap_name')):
            
            try:
                self.comp1_fractions_layer.events.colormap.disconnect(self._on_colormap_changed)
                self.comp1_fractions_layer.events.colormap.disconnect(self._sync_colormaps)
                self.comp1_fractions_layer.events.contrast_limits.disconnect(self._on_contrast_limits_changed)
                
                if self.comp2_fractions_layer is not None:
                    self.comp2_fractions_layer.events.colormap.disconnect(self._sync_colormaps)
                
                if self._saved_colormap_colors is not None:
                    from napari.utils.colormaps import Colormap
                    
                    if isinstance(self._saved_colormap_colors, list):
                        saved_colors = np.array(self._saved_colormap_colors)
                    else:
                        saved_colors = self._saved_colormap_colors
                    
                    saved_colormap = Colormap(colors=saved_colors, name="saved_custom")
                    self.comp1_fractions_layer.colormap = saved_colormap
                    
                    if self.comp2_fractions_layer is not None:
                        inverted_colors = saved_colors[::-1]
                        inverted_colormap = Colormap(colors=inverted_colors, name="saved_custom_inverted")
                        self.comp2_fractions_layer.colormap = inverted_colormap
                else:
                    self.comp1_fractions_layer.colormap = self._saved_colormap_name
                    if self.comp2_fractions_layer is not None:
                        inverted_name = self._saved_colormap_name + '_r' if not self._saved_colormap_name.endswith('_r') else self._saved_colormap_name[:-2]
                        self.comp2_fractions_layer.colormap = inverted_name
                
                if isinstance(self._saved_contrast_limits, list):
                    saved_limits = tuple(self._saved_contrast_limits)
                else:
                    saved_limits = self._saved_contrast_limits
                
                self.comp1_fractions_layer.contrast_limits = saved_limits
                if self.comp2_fractions_layer is not None:
                    self.comp2_fractions_layer.contrast_limits = saved_limits
                
                self.fractions_colormap = self.comp1_fractions_layer.colormap.colors
                self.colormap_contrast_limits = self.comp1_fractions_layer.contrast_limits
                
                self.comp1_fractions_layer.events.colormap.connect(self._on_colormap_changed)
                self.comp1_fractions_layer.events.colormap.connect(self._sync_colormaps)
                self.comp1_fractions_layer.events.contrast_limits.connect(self._on_contrast_limits_changed)
                
                if self.comp2_fractions_layer is not None:
                    self.comp2_fractions_layer.events.colormap.connect(self._sync_colormaps)
                
                self.draw_line_between_components()
                
            except Exception as e:
                print(f"Error applying saved colormap settings: {e}")
                try:
                    self.comp1_fractions_layer.events.colormap.connect(self._on_colormap_changed)
                    self.comp1_fractions_layer.events.colormap.connect(self._sync_colormaps)
                    self.comp1_fractions_layer.events.contrast_limits.connect(self._on_contrast_limits_changed)
                    if self.comp2_fractions_layer is not None:
                        self.comp2_fractions_layer.events.colormap.connect(self._sync_colormaps)
                except Exception:
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
        self._update_components_setting_in_metadata(
            'two_component_line_settings.line_offset', self.line_offset
        )

        if hasattr(self, 'line_offset_value_label'):
            self.line_offset_value_label.setText(f"{self.line_offset:.3f}")
        self.draw_line_between_components()
        if self.parent_widget is not None:
            self.parent_widget.canvas_widget.canvas.draw_idle()

    def _on_line_width_changed(self, value):
        """Handle changes to line width from spinbox."""
        self.line_width = float(value)
        self._update_components_setting_in_metadata(
            'two_component_line_settings.line_width', self.line_width
        )

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
        self._update_components_setting_in_metadata(
            'two_component_line_settings.line_alpha', self.line_alpha
        )

        if hasattr(self, 'line_alpha_value_label'):
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
            self._update_components_setting_in_metadata(
                'two_components_label_settings.color', self.label_color
            )
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

        for i, comp in enumerate(self.components):
            if comp is not None:
                if (
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

        old_display_name = old_name if old_name else f"Component {idx + 1}"
        new_display_name = new_name if new_name else f"Component {idx + 1}"

        old_layer_name = f"{old_display_name} fractions: {self.current_image_layer_name}"
        new_layer_name = f"{new_display_name} fractions: {self.current_image_layer_name}"

        if (
            old_layer_name in self.viewer.layers
            and old_layer_name != new_layer_name
        ):
            layer_obj = self.viewer.layers[old_layer_name]
            layer_obj.name = new_layer_name

            if idx == 0:
                self.comp1_fractions_layer = layer_obj
            elif idx == 1:
                self.comp2_fractions_layer = layer_obj

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
                    elif idx == 1:
                        self.comp2_fractions_layer = layer_obj
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

            layer = self.viewer.layers[self.current_image_layer_name]
            if (
                'settings' not in layer.metadata
                or 'component_analysis' not in layer.metadata['settings']
            ):
                return component_g, component_s, component_names

            settings = layer.metadata['settings']['component_analysis']
            components_data = settings.get('components', {})
            harmonic_key = str(harmonic)

            sorted_indices = sorted([int(k) for k in components_data.keys()])

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
        comp.g_edit.setText(f"{x:.3f}")
        comp.s_edit.setText(f"{y:.3f}")

        if comp.lifetime_edit is not None:
            comp.lifetime_edit.clear()

        current_harmonic = getattr(self.parent_widget, 'harmonic', 1)
        self._update_component_gs_coords(idx, current_harmonic, x, y)
        self._update_component_lifetime(idx, current_harmonic, None)

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
        """Handle changes to colormap of any fraction layer (linear projection or component fit)."""
        layer = event.source

        if (
            self.comp1_fractions_layer is not None
            and layer == self.comp1_fractions_layer
        ):
            self.fractions_colormap = layer.colormap.colors
            self.colormap_contrast_limits = layer.contrast_limits

        comp_idx = self._find_component_index_for_layer(layer)
        if comp_idx is None:
            return

        colormap_name = getattr(layer.colormap, 'name', 'custom')
        is_standard_colormap = self._is_standard_colormap(colormap_name)

        if is_standard_colormap:
            colormap_colors = None
        else:
            colormap_colors = layer.colormap.colors
            if colormap_colors is not None:
                if hasattr(colormap_colors, 'tolist'):
                    colormap_colors = colormap_colors.tolist()
                elif isinstance(colormap_colors, np.ndarray):
                    colormap_colors = colormap_colors.tolist()

        current_harmonic = getattr(self.parent_widget, 'harmonic', 1)
        self._update_component_colormap(
            comp_idx,
            current_harmonic,
            colormap_name if is_standard_colormap else None,
            colormap_colors,
            tuple(layer.contrast_limits),
        )

        self._update_component_colors()
        self.draw_line_between_components()

    def _on_contrast_limits_changed(self, event):
        """Handle changes to contrast limits of any fraction layer."""
        layer = event.source

        if (
            self.comp1_fractions_layer is not None
            and layer == self.comp1_fractions_layer
        ):
            self.colormap_contrast_limits = layer.contrast_limits

        comp_idx = self._find_component_index_for_layer(layer)
        if comp_idx is None:
            return

        contrast_limits = layer.contrast_limits
        if hasattr(contrast_limits, 'tolist'):
            contrast_limits = contrast_limits.tolist()
        elif isinstance(contrast_limits, np.ndarray):
            contrast_limits = contrast_limits.tolist()
        else:
            contrast_limits = list(contrast_limits)

        current_harmonic = getattr(self.parent_widget, 'harmonic', 1)

        if not self._updating_settings and self.current_image_layer_name:
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

        self.draw_line_between_components()

    def _find_component_index_for_layer(self, layer):
        """Find which component index a layer belongs to based on its name."""
        if not self.current_image_layer_name:
            return None

        layer_name = layer.name

        for i, comp in enumerate(self.components):
            if comp is not None:
                name = comp.name_edit.text().strip() or f"Component {i + 1}"

                expected_names = [
                    f"{name} fractions: {self.current_image_layer_name}",  # Linear projection
                    f"{name} fraction: {self.current_image_layer_name}",  # Component fit
                ]

                if layer_name in expected_names:
                    return i

        return None

    def _is_standard_colormap(self, colormap_name):
        """Check if a colormap name refers to a standard matplotlib/vispy/napari colormap."""
        try:
            import matplotlib.pyplot as plt

            plt.get_cmap(colormap_name)
            return True
        except Exception:
            pass

        try:
            import vispy.color

            vispy.color.get_colormap(colormap_name)
            return True
        except Exception:
            pass

        try:
            from napari.utils.colormaps import AVAILABLE_COLORMAPS

            if colormap_name in AVAILABLE_COLORMAPS:
                return True
        except Exception:
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
        comp.g_edit.setText(f"{x:.3f}")
        comp.s_edit.setText(f"{y:.3f}")

        if comp.lifetime_edit is not None:
            comp.lifetime_edit.clear()

        current_harmonic = getattr(self.parent_widget, 'harmonic', 1)
        self._update_component_gs_coords(
            self.dragging_component_idx, current_harmonic, x, y
        )
        self._update_component_lifetime(
            self.dragging_component_idx, current_harmonic, None
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
            layer = self.viewer.layers[self.current_image_layer_name]
            if (
                'settings' in layer.metadata
                and 'component_analysis' in layer.metadata['settings']
            ):

                settings = layer.metadata['settings']['component_analysis']
                components_data = settings.get('components', {})

                for comp_data in components_data.values():
                    gs_harmonics = comp_data.get('gs_harmonics', {})
                    for harmonic_str in gs_harmonics.keys():
                        harmonics.add(int(harmonic_str))

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
                    import matplotlib.pyplot as plt

                    mpl_cmap = plt.get_cmap(inverted_name)
                    colors = mpl_cmap(np.linspace(0, 1, 256))
                    return Colormap(colors=colors, name=inverted_name)
                except Exception:
                    if inverted_name in AVAILABLE_COLORMAPS:
                        return inverted_name

            try:
                import matplotlib.pyplot as plt

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
            except Exception:
                pass

        if colormap_colors is not None:
            if isinstance(colormap_colors, list):
                colormap_colors = np.array(colormap_colors)
            inverted_colors = colormap_colors[::-1]
            return Colormap(
                colors=inverted_colors, name=f"inverted_{colormap_name}"
            )

        return 'PiYG_r' if not colormap_name.endswith('_r') else 'PiYG'

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

                self.comp2_fractions_layer.events.colormap.disconnect(
                    self._sync_colormaps
                )
                inverted_colormap = self._get_inverted_colormap(
                    current_colormap
                )
                self.comp2_fractions_layer.colormap = inverted_colormap
                self.comp2_fractions_layer.events.colormap.connect(
                    self._sync_colormaps
                )

            elif event.source == self.comp2_fractions_layer:
                current_colormap = self.comp2_fractions_layer.colormap

                self.comp1_fractions_layer.events.colormap.disconnect(
                    self._sync_colormaps
                )
                inverted_colormap = self._get_inverted_colormap(
                    current_colormap
                )
                self.comp1_fractions_layer.colormap = inverted_colormap
                self.comp1_fractions_layer.events.colormap.connect(
                    self._sync_colormaps
                )

            if hasattr(self.comp1_fractions_layer.colormap, 'colors'):
                self.fractions_colormap = (
                    self.comp1_fractions_layer.colormap.colors
                )

            self.draw_line_between_components()

        except Exception as e:
            print(f"Error in _sync_colormaps: {e}")

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
                self.comp1_fractions_layer.events.colormap.connect(
                    self._sync_colormaps
                )
                self.comp1_fractions_layer.events.contrast_limits.connect(
                    self._on_contrast_limits_changed
                )
            elif idx == 1:
                self.comp2_fractions_layer = self.viewer.layers[expected_name]
                self.comp2_fractions_layer.events.colormap.connect(
                    self._sync_colormaps
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
                        self.comp1_fractions_layer.events.colormap.connect(
                            self._sync_colormaps
                        )
                        self.comp1_fractions_layer.events.contrast_limits.connect(
                            self._on_contrast_limits_changed
                        )
                    elif idx == 1:
                        self.comp2_fractions_layer = layer_obj
                        self.comp2_fractions_layer.events.colormap.connect(
                            self._sync_colormaps
                        )
                    break

        if (
            self.comp1_fractions_layer is not None
            and self.comp2_fractions_layer is not None
            and idx == 1
        ):
            try:
                link_layers(
                    [self.comp1_fractions_layer, self.comp2_fractions_layer],
                    ('contrast_limits', 'gamma'),
                )
            except Exception:
                pass

    def _reconnect_existing_fraction_layers(self, layer_name):
        """Reconnect to existing fraction layers if they exist."""
        layer = self.viewer.layers[layer_name]

        comp1_name = "Component 1"
        comp2_name = "Component 2"

        if (
            'settings' in layer.metadata
            and 'component_analysis' in layer.metadata['settings']
        ):

            settings = layer.metadata['settings']['component_analysis']

            if 'components' in settings and len(settings['components']) > 0:
                if '0' in settings['components']:
                    comp1_name = (
                        settings['components']['0'].get('name')
                        or "Component 1"
                    )
                if '1' in settings['components']:
                    comp2_name = (
                        settings['components']['1'].get('name')
                        or "Component 2"
                    )

        comp1_fractions_layer_name = f"{comp1_name} fractions: {layer_name}"
        comp2_fractions_layer_name = f"{comp2_name} fractions: {layer_name}"

        self._find_and_reconnect_layer(
            comp1_fractions_layer_name, comp1_name, layer_name, 0
        )
        self._find_and_reconnect_layer(
            comp2_fractions_layer_name, comp2_name, layer_name, 1
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
        self.current_image_layer_name = (
            self.parent_widget.image_layer_with_phasor_features_combobox.currentText()
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

        if self.comp1_fractions_layer is not None:
            try:
                self.comp1_fractions_layer.events.colormap.disconnect(
                    self._on_colormap_changed
                )
                self.comp1_fractions_layer.events.colormap.disconnect(
                    self._sync_colormaps
                )
                self.comp1_fractions_layer.events.contrast_limits.disconnect(
                    self._on_contrast_limits_changed
                )
            except Exception:
                pass

        if self.comp2_fractions_layer is not None:
            try:
                self.comp2_fractions_layer.events.colormap.disconnect(
                    self._sync_colormaps
                )
            except Exception:
                pass

        self.comp1_fractions_layer = None
        self.comp2_fractions_layer = None
        self.fractions_colormap = None
        self.colormap_contrast_limits = None

        self._update_lifetime_inputs_visibility()

        layer_name = (
            self.parent_widget.image_layer_with_phasor_features_combobox.currentText()
        )
        if layer_name:
            self._reconnect_existing_fraction_layers(layer_name)

            self._restore_and_recreate_components_from_metadata()

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

        if not self._updating_settings and self.current_image_layer_name:
            layer = self.viewer.layers[self.current_image_layer_name]
            settings = layer.metadata.get('settings', {}).get(
                'component_analysis', {}
            )
            current_harmonic = getattr(self.parent_widget, 'harmonic', 1)
            harmonic_key = str(current_harmonic)

        for i, (fraction, name) in enumerate(zip(fractions, component_names)):
            fraction_reshaped = fraction.reshape(
                self.parent_widget._labels_layer_with_phasor_features.data.shape
            )

            layer_name = f"{name} fraction: {base_name}"

            colormap = None
            contrast_limits = (0, 1)
            idx_str = str(i)

            if (
                not self._updating_settings
                and self.current_image_layer_name
                and idx_str in settings.get('components', {})
                and harmonic_key
                in settings['components'][idx_str].get('gs_harmonics', {})
            ):

                harmonic_data = settings['components'][idx_str][
                    'gs_harmonics'
                ][harmonic_key]

                if harmonic_data.get('colormap_name'):
                    colormap = harmonic_data['colormap_name']
                elif harmonic_data.get('colormap_colors'):
                    from napari.utils.colormaps import Colormap

                    colors = harmonic_data['colormap_colors']
                    if isinstance(colors, list):
                        colors = np.array(colors)
                    colormap = Colormap(colors=colors, name="saved_custom")

                if harmonic_data.get('contrast_limits'):
                    contrast_limits = tuple(harmonic_data['contrast_limits'])

            if colormap is None:
                if i < len(self.component_colormap_names):
                    colormap = self.component_colormap_names[i]
                else:
                    colormap = 'viridis'

            layer = self.viewer.add_image(
                fraction_reshaped,
                name=layer_name,
                scale=self.parent_widget._labels_layer_with_phasor_features.scale,
                colormap=colormap,
                contrast_limits=contrast_limits,
            )

            self.fraction_layers.append(layer)
            layer.events.colormap.connect(self._on_colormap_changed)

        self._update_component_colors()

    def _ensure_component_metadata(self, idx: int, harmonic: int = None):
        """Ensure component metadata structure exists and return component data dict."""
        if self._updating_settings or not self.current_image_layer_name:
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
            }
        )

    def _run_analysis(self):
        """Run the selected analysis and store component locations in metadata."""
        self._analysis_attempted = True
        self._update_all_component_styling()

        if not self._updating_settings and self.current_image_layer_name:
            layer = self.viewer.layers[self.current_image_layer_name]
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
        comp1_fractions_layer_name = (
            f"{comp1_name} fractions: {self.current_image_layer_name}"
        )
        comp2_name = c2.name_edit.text().strip() or "Component 2"
        comp2_fractions_layer_name = (
            f"{comp2_name} fractions: {self.current_image_layer_name}"
        )

        layer = self.viewer.layers[self.current_image_layer_name]
        settings = layer.metadata.get('settings', {}).get(
            'component_analysis', {}
        )
        current_harmonic = getattr(self.parent_widget, 'harmonic', 1)
        harmonic_key = str(current_harmonic)

        comp1_colormap = 'PiYG'
        contrast_limits = (0, 1)

        if '0' in settings.get('components', {}):
            comp_data = settings['components']['0']
            if harmonic_key in comp_data.get('gs_harmonics', {}):
                harmonic_data = comp_data['gs_harmonics'][harmonic_key]

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
                    contrast_limits = tuple(harmonic_data['contrast_limits'])

        if comp1_fractions_layer_name in self.viewer.layers:
            self.viewer.layers.remove(
                self.viewer.layers[comp1_fractions_layer_name]
            )
        if comp2_fractions_layer_name in self.viewer.layers:
            self.viewer.layers.remove(
                self.viewer.layers[comp2_fractions_layer_name]
            )

        comp1_selected_fractions_layer = Image(
            fractions,
            name=comp1_fractions_layer_name,
            scale=self.parent_widget._labels_layer_with_phasor_features.scale,
            colormap=comp1_colormap,
            contrast_limits=contrast_limits,
        )

        self.comp1_fractions_layer = self.viewer.add_layer(
            comp1_selected_fractions_layer
        )

        comp2_colormap = self._get_inverted_colormap(
            self.comp1_fractions_layer.colormap
        )

        comp2_selected_fractions_layer = Image(
            1.0 - fractions,
            name=comp2_fractions_layer_name,
            scale=self.parent_widget._labels_layer_with_phasor_features.scale,
            colormap=comp2_colormap,
            contrast_limits=contrast_limits,
        )

        self.comp2_fractions_layer = self.viewer.add_layer(
            comp2_selected_fractions_layer
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
                    if colormap_colors is not None:
                        if hasattr(colormap_colors, 'tolist'):
                            colormap_colors = colormap_colors.tolist()
                        elif isinstance(colormap_colors, np.ndarray):
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

        self.comp1_fractions_layer.events.colormap.connect(
            self._on_colormap_changed
        )
        self.comp1_fractions_layer.events.colormap.connect(
            self._sync_colormaps
        )
        self.comp2_fractions_layer.events.colormap.connect(
            self._sync_colormaps
        )
        self.comp1_fractions_layer.events.contrast_limits.connect(
            self._on_contrast_limits_changed
        )

        link_layers(
            [self.comp1_fractions_layer, self.comp2_fractions_layer],
            ('contrast_limits', 'gamma'),
        )

        self._update_component_colors()
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

            mean = self.viewer.layers[self.current_image_layer_name].metadata[
                'original_mean'
            ]

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

            phasor_data = (
                self.parent_widget._labels_layer_with_phasor_features.features
            )

            mean = self.viewer.layers[self.current_image_layer_name].metadata[
                'original_mean'
            ]

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

            real = np.stack(real_list, axis=0)
            imag = np.stack(imag_list, axis=0)

        try:
            fractions = phasor_component_fit(
                mean, real, imag, component_g, component_s
            )

            layer = self.viewer.layers[self.current_image_layer_name]
            settings = layer.metadata['settings']['component_analysis']
            harmonic_key = str(current_harmonic)

            self.fraction_layers.clear()

            for i, (fraction, name) in enumerate(
                zip(fractions, component_names)
            ):
                fraction_reshaped = fraction.reshape(
                    self.parent_widget._labels_layer_with_phasor_features.data.shape
                )

                fraction_layer_name = f"{name} fraction: {self.current_image_layer_name}"

                colormap = None
                contrast_limits = (0, 1)
                idx_str = str(i)

                if (
                    not self._updating_settings
                    and idx_str in settings.get('components', {})
                    and harmonic_key
                    in settings['components'][idx_str].get('gs_harmonics', {})
                ):

                    harmonic_data = settings['components'][idx_str][
                        'gs_harmonics'
                    ][harmonic_key]

                    if harmonic_data.get('colormap_name'):
                        colormap = harmonic_data['colormap_name']
                    elif harmonic_data.get('colormap_colors'):
                        from napari.utils.colormaps import Colormap

                        colors = harmonic_data['colormap_colors']
                        if isinstance(colors, list):
                            colors = np.array(colors)
                        colormap = Colormap(colors=colors, name="saved_custom")

                    if harmonic_data.get('contrast_limits'):
                        contrast_limits = tuple(
                            harmonic_data['contrast_limits']
                        )

                if colormap is None and fraction_layer_name in self.viewer.layers:
                    existing_layer = self.viewer.layers[fraction_layer_name]
                    colormap = existing_layer.colormap
                    contrast_limits = existing_layer.contrast_limits

                if colormap is None:
                    if i < len(self.component_colormap_names):
                        colormap = self.component_colormap_names[i]
                    else:
                        colormap = 'viridis'

                # Remove previous layer if it exists (avoids duplication)
                try:
                    self.viewer.layers.remove(
                        self.viewer.layers[fraction_layer_name]
                    )
                except KeyError:
                    pass
                new_layer = self.viewer.add_image(
                    fraction_reshaped,
                    name=fraction_layer_name,
                    scale=self.parent_widget._labels_layer_with_phasor_features.scale,
                    colormap=colormap,
                )

                new_layer.contrast_limits = contrast_limits

                self.fraction_layers.append(new_layer)
                new_layer.events.colormap.connect(self._on_colormap_changed)

                if not self._updating_settings:
                    if idx_str in settings['components']:
                        comp_data = settings['components'][idx_str]
                        if harmonic_key not in comp_data['gs_harmonics']:
                            comp_data['gs_harmonics'][harmonic_key] = {}

                        colormap_name = getattr(
                            new_layer.colormap, 'name', 'custom'
                        )
                        is_standard_colormap = False
                        try:
                            import matplotlib.pyplot as plt

                            plt.get_cmap(colormap_name)
                            is_standard_colormap = True
                        except Exception:
                            try:
                                import vispy.color

                                vispy.color.get_colormap(colormap_name)
                                is_standard_colormap = True
                            except Exception:
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
                            if colormap_colors is not None:
                                if hasattr(colormap_colors, 'tolist'):
                                    colormap_colors = colormap_colors.tolist()
                                elif isinstance(colormap_colors, np.ndarray):
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

            self._update_component_colors()

        except Exception as e:
            show_error(f"Analysis failed: {str(e)}")
