from dataclasses import dataclass
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from napari.experimental import link_layers
from napari.layers import Image
from napari.utils.notifications import show_error
from phasorpy.component import phasor_component_fraction
from phasorpy.lifetime import phasor_from_lifetime
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QColorDialog,
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
        self.comp1_fractions_layer = None
        self.comp2_fractions_layer = None  # Add this missing attribute
        self.fractions_colormap = None
        self.colormap_contrast_limits = None

        # Style state
        self.label_fontsize = 10
        self.label_bold = False
        self.label_italic = False
        self.label_color = 'black'

        # Line settings
        self.show_colormap_line = True
        self.show_component_dots = True
        self.line_offset = 0.0
        self.line_width = 3.0
        self.line_alpha = 1
        self.default_component_color = 'dimgray'

        # Flag to prevent clearing lifetime when updating from lifetime
        self._updating_from_lifetime = False

        # Dialog / event flags
        self.plot_dialog = None
        self.style_dialog = None
        self.drag_events_connected = False

        # Drag state
        self.dragging_component_idx = None
        self.dragging_label_idx = None

        self.setup_ui()

        # Connect to layer selection change to show/hide lifetime inputs
        if hasattr(
            self.parent_widget, 'image_layer_with_phasor_features_combobox'
        ):
            self.parent_widget.image_layer_with_phasor_features_combobox.currentTextChanged.connect(
                self._update_lifetime_inputs_visibility
            )

        self._update_lifetime_inputs_visibility()

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

        # Component 1 section
        comp1_layout = QHBoxLayout()
        comp1_layout.addWidget(QLabel("Component 1:"))
        self.comp1_name_edit = QLineEdit()
        self.comp1_name_edit.setPlaceholderText("Component name (optional)")
        comp1_layout.addWidget(self.comp1_name_edit)
        self.first_button = QPushButton("Select Component 1")
        comp1_layout.addWidget(self.first_button)

        # Component 1 coordinates
        coord1_layout = QHBoxLayout()
        self.first_lifetime_label = QLabel("τ:")
        self.first_lifetime_edit = QLineEdit()
        self.first_lifetime_edit.setPlaceholderText(
            "Lifetime in ns (optional)"
        )
        coord1_layout.addWidget(self.first_lifetime_label)
        coord1_layout.addWidget(self.first_lifetime_edit)

        coord1_layout.addWidget(QLabel("G:"))
        self.first_edit1 = QLineEdit()
        self.first_edit1.setPlaceholderText("Real coordinate")
        coord1_layout.addWidget(self.first_edit1)
        coord1_layout.addWidget(QLabel("S:"))
        self.first_edit2 = QLineEdit()
        self.first_edit2.setPlaceholderText("Imaginary coordinate")
        coord1_layout.addWidget(self.first_edit2)

        # Component 2 section
        comp2_layout = QHBoxLayout()
        comp2_layout.addWidget(QLabel("Component 2:"))
        self.comp2_name_edit = QLineEdit()
        self.comp2_name_edit.setPlaceholderText("Component name (optional)")
        comp2_layout.addWidget(self.comp2_name_edit)
        self.second_button = QPushButton("Select Component 2")
        comp2_layout.addWidget(self.second_button)

        # Component 2 coordinates
        coord2_layout = QHBoxLayout()
        self.second_lifetime_label = QLabel("τ:")
        self.second_lifetime_edit = QLineEdit()
        self.second_lifetime_edit.setPlaceholderText(
            "Lifetime in ns (optional)"
        )
        coord2_layout.addWidget(self.second_lifetime_label)
        coord2_layout.addWidget(self.second_lifetime_edit)

        coord2_layout.addWidget(QLabel("G:"))
        self.second_edit1 = QLineEdit()
        self.second_edit1.setPlaceholderText("Real coordinate")
        coord2_layout.addWidget(self.second_edit1)
        coord2_layout.addWidget(QLabel("S:"))
        self.second_edit2 = QLineEdit()
        self.second_edit2.setPlaceholderText("Imaginary coordinate")
        coord2_layout.addWidget(self.second_edit2)

        # Calculate button
        self.calculate_button = QPushButton(
            "Display Component Fraction Images"
        )
        self.calculate_button.clicked.connect(self.on_calculate_button_clicked)

        # Add all layouts to content layout
        layout.addLayout(comp1_layout)
        layout.addLayout(coord1_layout)
        layout.addLayout(comp2_layout)
        layout.addLayout(coord2_layout)

        # Plot settings section
        buttons_row = QHBoxLayout()
        self.plot_settings_btn = QPushButton("Edit Line Layout...")
        self.plot_settings_btn.clicked.connect(self._open_plot_settings_dialog)
        buttons_row.addWidget(self.plot_settings_btn)

        # Label style button
        self.label_style_btn = QPushButton("Edit Component Name Layout...")
        self.label_style_btn.clicked.connect(self._open_label_style_dialog)
        buttons_row.addWidget(self.label_style_btn)

        buttons_row.addStretch()
        layout.addLayout(buttons_row)

        layout.addWidget(self.calculate_button)
        layout.addStretch()

        comp1 = ComponentState(
            idx=0,
            name_edit=self.comp1_name_edit,
            lifetime_edit=self.first_lifetime_edit,
            g_edit=self.first_edit1,
            s_edit=self.first_edit2,
            select_button=self.first_button,
            label="Component 1",
            text_offset=(0.02, 0.02),
        )
        comp2 = ComponentState(
            idx=1,
            name_edit=self.comp2_name_edit,
            lifetime_edit=self.second_lifetime_edit,
            g_edit=self.second_edit1,
            s_edit=self.second_edit2,
            select_button=self.second_button,
            label="Component 2",
            text_offset=(0.02, 0.02),
        )
        self.components = [comp1, comp2]

        # Connect generic signals
        comp1.name_edit.textChanged.connect(
            lambda: self._on_component_name_changed(0)
        )
        comp2.name_edit.textChanged.connect(
            lambda: self._on_component_name_changed(1)
        )
        comp1.g_edit.editingFinished.connect(
            lambda: self._on_component_coords_changed(0)
        )
        comp1.s_edit.editingFinished.connect(
            lambda: self._on_component_coords_changed(0)
        )
        comp2.g_edit.editingFinished.connect(
            lambda: self._on_component_coords_changed(1)
        )
        comp2.s_edit.editingFinished.connect(
            lambda: self._on_component_coords_changed(1)
        )
        comp1.lifetime_edit.editingFinished.connect(
            lambda: self._update_component_from_lifetime(0)
        )
        comp2.lifetime_edit.editingFinished.connect(
            lambda: self._update_component_from_lifetime(1)
        )
        comp1.select_button.clicked.connect(lambda: self._select_component(0))
        comp2.select_button.clicked.connect(lambda: self._select_component(1))

        self.setLayout(root_layout)

    def _open_plot_settings_dialog(self):
        if self.plot_dialog is not None and self.plot_dialog.isVisible():
            self.plot_dialog.raise_()
            self.plot_dialog.activateWindow()
            return

        self.plot_dialog = QDialog(self)
        self.plot_dialog.setWindowTitle("Component Line Settings")
        vbox = QVBoxLayout(self.plot_dialog)

        # Row 1: checkboxes
        row1 = QHBoxLayout()
        self.colormap_line_checkbox = QCheckBox("Overlay Colormap to line")
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

    def _open_label_style_dialog(self):
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

        self.show_colormap_line = default_show_colormap_line
        self.show_component_dots = default_show_component_dots
        self.line_offset = default_line_offset
        self.line_width = default_line_width
        self.line_alpha = default_line_alpha

        self.colormap_line_checkbox.setChecked(default_show_colormap_line)
        self.show_dots_checkbox.setChecked(default_show_component_dots)

        self.line_offset_slider.setValue(int(default_line_offset * 1000))
        self.line_offset_value_label.setText(f"{default_line_offset:.3f}")

        self.line_width_spin.setValue(default_line_width)

        self.line_alpha_slider.setValue(int(default_line_alpha * 100))
        self.line_alpha_value_label.setText(f"{default_line_alpha:.2f}")

        # Reset component dots alpha
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
        artists = []
        for comp in self.components:
            if comp.dot is not None:
                artists.append(comp.dot)
            if comp.text is not None:
                artists.append(comp.text)
        if self.component_line is not None:
            artists.append(self.component_line)
        return artists

    def set_artists_visible(self, visible):
        """Set visibility of all artists created by this widget."""
        for comp in self.components:
            if comp.dot is not None:
                comp.dot.set_visible(visible and self.show_component_dots)
            if comp.text is not None:
                comp.text.set_visible(visible)
        if self.component_line is not None:
            self.component_line.set_visible(visible)

    def _toggle_plot_section(self, checked):
        self.plot_section.setVisible(checked)

    def _on_plot_setting_changed(self):
        self.show_colormap_line = self.colormap_line_checkbox.isChecked()
        self.show_component_dots = self.show_dots_checkbox.isChecked()

        if self.show_colormap_line and self.fractions_colormap is not None:
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
        self.line_offset = value / 1000.0
        self.line_offset_value_label.setText(f"{self.line_offset:.3f}")
        self.draw_line_between_components()
        if self.parent_widget is not None:
            self.parent_widget.canvas_widget.canvas.draw_idle()

    def _on_line_width_changed(self, value):
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
        self.style_section.setVisible(checked)

    def _pick_label_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.label_color = color.name()
            self._apply_styles_to_labels()

    def _on_label_style_changed(self):
        self.label_fontsize = self.fontsize_spin.value()
        self.label_bold = self.bold_checkbox.isChecked()
        self.label_italic = self.italic_checkbox.isChecked()
        self._apply_styles_to_labels()

    def _apply_styles_to_labels(self):
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
        has_freq = self.parent_widget._get_frequency_from_layer() is not None

        for w in (
            self.first_lifetime_label,
            self.first_lifetime_edit,
            self.second_lifetime_label,
            self.second_lifetime_edit,
        ):
            w.setVisible(has_freq)

        if has_freq:
            for i, comp in enumerate(self.components):
                if comp.lifetime_edit.text().strip():
                    self._update_component_from_lifetime(i)

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

    def _create_component_at_coordinates(self, idx: int, x: float, y: float):
        if self.parent_widget is None:
            return
        comp = self.components[idx]

        if self.show_colormap_line and self.fractions_colormap is not None:
            c1_color, c2_color = self._get_component_colors()
            color = c1_color if idx == 0 else c2_color
        else:
            color = self.default_component_color

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
            )
        self._make_components_draggable()
        self.parent_widget.canvas_widget.canvas.draw_idle()
        self.draw_line_between_components()

    def _on_component_name_changed(self, idx: int):
        comp = self.components[idx]
        if comp.dot is None:
            return
        name = comp.name_edit.text().strip()
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

    def _select_component(self, idx: int):
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
        self._redraw(force=True)

    def _get_component_colors(self):
        """Get colors for components based on the colormap ends."""
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
                # Normalize the fraction values to colormap indices
                component1_idx = int(
                    ((1.0 - vmin) / (vmax - vmin))
                    * (len(self.fractions_colormap) - 1)
                )
                component2_idx = int(
                    ((0.0 - vmin) / (vmax - vmin))
                    * (len(self.fractions_colormap) - 1)
                )

                # Clamp indices to valid range
                component1_idx = max(
                    0, min(len(self.fractions_colormap) - 1, component1_idx)
                )
                component2_idx = max(
                    0, min(len(self.fractions_colormap) - 1, component2_idx)
                )

                component1_color = self.fractions_colormap[component1_idx]
                component2_color = self.fractions_colormap[component2_idx]
            else:
                # Fallback if vmax <= vmin
                component1_color = self.fractions_colormap[-1]
                component2_color = self.fractions_colormap[0]

            return component1_color, component2_color
        else:
            # Default colors if no colormap is available
            return 'red', 'blue'

    def _update_component_colors(self):
        """Update the colors of the component dots to match colormap ends."""
        c1_color, c2_color = self._get_component_colors()
        if len(self.components) > 0 and self.components[0].dot is not None:
            self.components[0].dot.set_color(c1_color)
        if len(self.components) > 1 and self.components[1].dot is not None:
            self.components[1].dot.set_color(c2_color)
        if self.parent_widget is not None:
            self.parent_widget.canvas_widget.canvas.draw_idle()

    def draw_line_between_components(self):
        """Draw a line between the first two components if both exist."""
        if len(self.components) < 2:
            return
        if not all(c.dot is not None for c in self.components[:2]):
            return

        if self.component_line is not None:
            try:
                self.component_line.remove()
            except (ValueError, AttributeError):
                pass
            self.component_line = None

        try:
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
            else:
                self.component_line = ax.plot(
                    [ox1, ox2],
                    [oy1, oy2],
                    color=self.default_component_color,
                    linewidth=self.line_width,
                    alpha=self.line_alpha,
                )[0]
                if hasattr(self.component_line, "set_solid_capstyle"):
                    try:
                        self.component_line.set_solid_capstyle('butt')
                    except Exception:
                        pass
                for comp in self.components:
                    if comp.dot is not None:
                        comp.dot.set_color(self.default_component_color)

            self.parent_widget.canvas_widget.canvas.draw_idle()

            components_tab_is_active = (
                self.parent_widget is not None
                and getattr(self.parent_widget, "tab_widget", None) is not None
                and self.parent_widget.tab_widget.currentWidget()
                is self.parent_widget.components_tab
            )
            self.set_artists_visible(components_tab_is_active)

        except Exception as e:
            show_error(f"Error drawing line: {str(e)}")

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

        # Build a continuous colormap
        if self.fractions_colormap is not None:
            if len(self.fractions_colormap) <= 32:
                # Create a smooth interpolated cmap from sparse control points
                colormap = LinearSegmentedColormap.from_list(
                    "fractions_interp", self.fractions_colormap, N=256
                )
            else:
                colormap = ListedColormap(self.fractions_colormap)
        else:
            colormap = plt.cm.PiYG

        # Get the actual contrast limits from the fractions layer
        if (
            hasattr(self, 'colormap_contrast_limits')
            and self.colormap_contrast_limits is not None
        ):
            vmin, vmax = self.colormap_contrast_limits
        elif self.comp1_fractions_layer is not None:
            vmin, vmax = self.comp1_fractions_layer.contrast_limits
        else:
            vmin, vmax = 0, 1

        # Create line segment
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
        """Make component dots draggable."""
        if self.parent_widget is None or self.drag_events_connected:
            return

        self.parent_widget.canvas_widget.canvas.mpl_connect(
            'button_press_event', self._on_press
        )
        self.parent_widget.canvas_widget.canvas.mpl_connect(
            'button_release_event', self._on_release
        )
        self.parent_widget.canvas_widget.canvas.mpl_connect(
            'motion_notify_event', self._on_motion
        )
        self.drag_events_connected = True

    def _on_press(self, event):
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

                    from napari.utils.colormaps import Colormap

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

                    from napari.utils.colormaps import Colormap

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

    def on_calculate_button_clicked(self):
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
