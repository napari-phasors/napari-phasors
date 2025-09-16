from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
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


class ComponentsWidget(QWidget):
    """Widget to perform component analysis on phasor coordinates."""

    def __init__(self, viewer: "napari.viewer.Viewer", parent=None):
        super().__init__()
        self.viewer = viewer
        self.parent_widget = parent

        # Initialize dot and line references
        self.component1_dot = None
        self.component2_dot = None
        self.component1_text = None
        self.component2_text = None
        self.component_line = None
        self.fractions_layer = None
        self.fractions_colormap = None
        self.colormap_contrast_limits = None
        self.plot_dialog = None
        self.style_dialog = None

        # Text offsets
        self.component1_text_offset = (0.02, 0.02)
        self.component2_text_offset = (0.02, 0.02)

        # Label style state
        self.label_fontsize = 10
        self.label_bold = False
        self.label_italic = False
        self.label_color = 'black'

        # Plot settings state
        self.show_colormap_line = True
        self.show_component_dots = True
        self.line_offset = 0.0
        self.line_width = 3.0
        self.line_alpha = 0.8

        # Dragging state
        self.dragging_component = None
        self.dragging_component_label = None
        self.press_event = None
        self.drag_events_connected = False

        # Setup UI
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
        self.comp1_name_edit.textChanged.connect(
            self._on_component1_name_changed
        )
        comp1_layout.addWidget(self.comp1_name_edit)
        self.my_button = QPushButton("Select Component 1")
        self.my_button.clicked.connect(self.on_first_button_clicked)
        comp1_layout.addWidget(self.my_button)

        # Component 1 coordinates
        coord1_layout = QHBoxLayout()
        self.first_lifetime_label = QLabel("τ:")
        self.first_lifetime_edit = QLineEdit()
        self.first_lifetime_edit.setPlaceholderText("Lifetime (optional)")
        self.first_lifetime_edit.editingFinished.connect(
            self._on_component1_lifetime_changed
        )
        coord1_layout.addWidget(self.first_lifetime_label)
        coord1_layout.addWidget(self.first_lifetime_edit)

        coord1_layout.addWidget(QLabel("G:"))
        self.first_edit1 = QLineEdit()
        self.first_edit1.setPlaceholderText("Real coordinate")
        self.first_edit1.editingFinished.connect(
            self._on_component1_coords_changed
        )
        coord1_layout.addWidget(self.first_edit1)
        coord1_layout.addWidget(QLabel("S:"))
        self.first_edit2 = QLineEdit()
        self.first_edit2.setPlaceholderText("Imaginary coordinate")
        self.first_edit2.editingFinished.connect(
            self._on_component1_coords_changed
        )
        coord1_layout.addWidget(self.first_edit2)

        # Component 2 section
        comp2_layout = QHBoxLayout()
        comp2_layout.addWidget(QLabel("Component 2:"))
        self.comp2_name_edit = QLineEdit()
        self.comp2_name_edit.setPlaceholderText("Component name (optional)")
        self.comp2_name_edit.textChanged.connect(
            self._on_component2_name_changed
        )
        comp2_layout.addWidget(self.comp2_name_edit)
        self.second_button = QPushButton("Select Component 2")
        self.second_button.clicked.connect(self.on_second_button_clicked)
        comp2_layout.addWidget(self.second_button)

        # Component 2 coordinates
        coord2_layout = QHBoxLayout()
        self.second_lifetime_label = QLabel("τ:")
        self.second_lifetime_edit = QLineEdit()
        self.second_lifetime_edit.setPlaceholderText("Lifetime (optional)")
        self.second_lifetime_edit.editingFinished.connect(
            self._on_component2_lifetime_changed
        )
        coord2_layout.addWidget(self.second_lifetime_label)
        coord2_layout.addWidget(self.second_lifetime_edit)

        coord2_layout.addWidget(QLabel("G:"))
        self.second_edit1 = QLineEdit()
        self.second_edit1.setPlaceholderText("Real coordinate")
        self.second_edit1.editingFinished.connect(
            self._on_component2_coords_changed
        )
        coord2_layout.addWidget(self.second_edit1)
        coord2_layout.addWidget(QLabel("S:"))
        self.second_edit2 = QLineEdit()
        self.second_edit2.setPlaceholderText("Imaginary coordinate")
        self.second_edit2.editingFinished.connect(
            self._on_component2_coords_changed
        )
        coord2_layout.addWidget(self.second_edit2)

        # Calculate button
        self.calculate_button = QPushButton("Analyze components fractions")
        self.calculate_button.clicked.connect(self.on_calculate_button_clicked)

        # Add all layouts to content layout
        layout.addLayout(comp1_layout)
        layout.addLayout(coord1_layout)
        layout.addLayout(comp2_layout)
        layout.addLayout(coord2_layout)

        # Plot settings section
        buttons_row = QHBoxLayout()
        self.plot_settings_btn = QPushButton("Component Line Settings")
        self.plot_settings_btn.clicked.connect(self._open_plot_settings_dialog)
        buttons_row.addWidget(self.plot_settings_btn)

        # Label style button
        self.label_style_btn = QPushButton("Component Label Style")
        self.label_style_btn.clicked.connect(self._open_label_style_dialog)
        buttons_row.addWidget(self.label_style_btn)

        # buttons_row.addStretch()
        layout.addLayout(buttons_row)

        layout.addWidget(self.calculate_button)
        layout.addStretch()

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

        # Close button
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.plot_dialog.close)
        vbox.addWidget(buttons)

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

    def get_all_artists(self):
        """Return a list of all matplotlib artists created by this widget."""
        artists = []
        if self.component1_dot is not None:
            artists.append(self.component1_dot)
        if self.component2_dot is not None:
            artists.append(self.component2_dot)
        if self.component1_text is not None:
            artists.append(self.component1_text)
        if self.component2_text is not None:
            artists.append(self.component2_text)
        if self.component_line is not None:
            artists.append(self.component_line)
        return artists

    def set_artists_visible(self, visible):
        """Set visibility of all artists created by this widget."""
        for artist in self.get_all_artists():
            if artist is None:
                continue
            if artist in (self.component1_dot, self.component2_dot):
                artist.set_visible(visible and self.show_component_dots)
            else:
                artist.set_visible(visible)

    def _toggle_plot_section(self, checked):
        self.plot_section.setVisible(checked)

    def _on_plot_setting_changed(self):
        self.show_colormap_line = self.colormap_line_checkbox.isChecked()
        self.show_component_dots = self.show_dots_checkbox.isChecked()

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
        for txt in (self.component1_text, self.component2_text):
            if txt is not None:
                txt.set_fontsize(self.label_fontsize)
                txt.set_fontweight(weight)
                txt.set_fontstyle(style)
                txt.set_color(self.label_color)
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

        # Also update lifetime-based G,S if frequency is now available
        if has_freq:
            self._on_component1_lifetime_changed()
            self._on_component2_lifetime_changed()

    def _compute_phasor_from_lifetime(self, lifetime_text):
        """Compute (G,S) from lifetime string; return tuple or (None,None)."""
        try:
            lifetime = float(lifetime_text)
        except (TypeError, ValueError):
            return None, None
        freq = self.parent_widget._get_frequency_from_layer()
        if freq is None:
            return None, None
        # Use current harmonic if available
        harmonic = getattr(self.parent_widget, "harmonic", 1)
        re, im = phasor_from_lifetime(freq * harmonic, lifetime)
        if np.ndim(re) > 0:
            re = float(np.array(re).ravel()[0])
        if np.ndim(im) > 0:
            im = float(np.array(im).ravel()[0])
        return re, im

    def _on_component1_lifetime_changed(self):
        """Update component 1 G,S from lifetime input."""
        if self.first_lifetime_edit.text().strip() == "":
            return
        G, S = self._compute_phasor_from_lifetime(
            self.first_lifetime_edit.text().strip()
        )
        if G is None:
            return
        self.first_edit1.setText(f"{G:.6f}")
        self.first_edit2.setText(f"{S:.6f}")
        self._on_component1_coords_changed()

    def _on_component2_lifetime_changed(self):
        """Update component 2 G,S from lifetime input."""
        if self.second_lifetime_edit.text().strip() == "":
            return
        G, S = self._compute_phasor_from_lifetime(
            self.second_lifetime_edit.text().strip()
        )
        if G is None:
            return
        self.second_edit1.setText(f"{G:.6f}")
        self.second_edit2.setText(f"{S:.6f}")
        self._on_component2_coords_changed()

    def _on_component1_coords_changed(self):
        """Handle changes to component 1 coordinates in line edits."""
        try:
            x = float(self.first_edit1.text())
            y = float(self.first_edit2.text())

            if self.component1_dot is not None:
                self.component1_dot.set_data([x], [y])
                if self.component1_text is not None:
                    ox, oy = self.component1_text_offset
                    self.component1_text.set_position((x + ox, y + oy))
                self.draw_line_between_components()
            else:
                self._create_component1_at_coordinates(x, y)
        except ValueError:
            pass

    def _on_component2_coords_changed(self):
        """Handle changes to component 2 coordinates in line edits."""
        try:
            x = float(self.second_edit1.text())
            y = float(self.second_edit2.text())

            if self.component2_dot is not None:
                self.component2_dot.set_data([x], [y])
                if self.component2_text is not None:
                    ox, oy = self.component2_text_offset
                    self.component2_text.set_position((x + ox, y + oy))
                self.draw_line_between_components()
            else:
                self._create_component2_at_coordinates(x, y)
        except ValueError:
            pass

    def _create_component1_at_coordinates(self, x, y):
        """Create component 1 at the specified coordinates."""
        if self.parent_widget is None:
            return

        component1_color, _ = self._get_component_colors()

        ax = self.parent_widget.canvas_widget.figure.gca()
        self.component1_dot = ax.plot(
            x,
            y,
            'o',
            color=component1_color,
            markersize=8,
            label='Component 1',
        )[0]

        name = self.comp1_name_edit.text().strip()
        if name:
            ox, oy = self.component1_text_offset
            self.component1_text = ax.text(
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

    def _create_component2_at_coordinates(self, x, y):
        """Create component 2 at the specified coordinates."""
        if self.parent_widget is None:
            return

        _, component2_color = self._get_component_colors()

        ax = self.parent_widget.canvas_widget.figure.gca()
        self.component2_dot = ax.plot(
            x,
            y,
            'o',
            color=component2_color,
            markersize=8,
            label='Component 2',
        )[0]

        name = self.comp2_name_edit.text().strip()
        if name:
            ox, oy = self.component2_text_offset
            self.component2_text = ax.text(
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

    def _on_component1_name_changed(self):
        """Handle changes to component 1 name."""
        if self.component1_dot is None:
            return
        name = self.comp1_name_edit.text().strip()
        prev_pos = None
        if self.component1_text is not None:
            prev_pos = self.component1_text.get_position()
            self.component1_text.remove()
            self.component1_text = None
        if name:
            dx, dy = self.component1_dot.get_data()
            if prev_pos is None:
                ox, oy = self.component1_text_offset
                base_x, base_y = dx[0] + ox, dy[0] + oy
            else:
                base_x, base_y = prev_pos
                self.component1_text_offset = (base_x - dx[0], base_y - dy[0])
            ax = self.parent_widget.canvas_widget.figure.gca()
            self.component1_text = ax.text(
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

    def _on_component2_name_changed(self):
        """Handle changes to component 2 name."""
        if self.component2_dot is None:
            return
        name = self.comp2_name_edit.text().strip()
        prev_pos = None
        if self.component2_text is not None:
            prev_pos = self.component2_text.get_position()
            self.component2_text.remove()
            self.component2_text = None
        if name:
            dx, dy = self.component2_dot.get_data()
            if prev_pos is None:
                ox, oy = self.component2_text_offset
                base_x, base_y = dx[0] + ox, dy[0] + oy
            else:
                base_x, base_y = prev_pos
                self.component2_text_offset = (base_x - dx[0], base_y - dy[0])
            ax = self.parent_widget.canvas_widget.figure.gca()
            self.component2_text = ax.text(
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
        component1_color, component2_color = self._get_component_colors()

        if self.component1_dot is not None:
            self.component1_dot.set_color(component1_color)

        if self.component2_dot is not None:
            self.component2_dot.set_color(component2_color)

        if self.parent_widget is not None:
            self.parent_widget.canvas_widget.canvas.draw_idle()

    def draw_line_between_components(self):
        """Draw a line between component 1 and component 2 if both exist."""
        if self.component1_dot is None or self.component2_dot is None:
            return
        if self.component_line is not None:
            try:
                self.component_line.remove()
            except (ValueError, AttributeError):
                pass
            self.component_line = None

        try:
            x1, y1 = self.component1_dot.get_data()
            x2, y2 = self.component2_dot.get_data()

            ox1, oy1, ox2, oy2 = x1[0], y1[0], x2[0], y2[0]
            if self.line_offset != 0.0:
                vx = x2[0] - x1[0]
                vy = y2[0] - y1[0]
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
                and self.fractions_layer is not None
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
                    color='k',
                    linewidth=self.line_width,
                    alpha=self.line_alpha,
                )[0]

                if hasattr(self.component_line, "set_solid_capstyle"):
                    try:
                        self.component_line.set_solid_capstyle('butt')
                    except Exception:
                        pass

            self.parent_widget.canvas_widget.canvas.draw_idle()

            # Only show line if this tab is active
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
            colormap = plt.cm.plasma

        # Get the actual contrast limits from the fractions layer
        if (
            hasattr(self, 'colormap_contrast_limits')
            and self.colormap_contrast_limits is not None
        ):
            vmin, vmax = self.colormap_contrast_limits
        elif self.fractions_layer is not None:
            vmin, vmax = self.fractions_layer.contrast_limits
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
            self.fractions_layer is not None
            and self.component_line is not None
        ):
            layer = event.source
            self.fractions_colormap = layer.colormap.colors
            self.colormap_contrast_limits = layer.contrast_limits

            self.draw_line_between_components()

    def _on_contrast_limits_changed(self, event):
        """Handle changes to the contrast limits of the fractions layer."""
        if (
            self.fractions_layer is not None
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
        """Handle mouse press for dragging components."""
        if event.inaxes is None:
            return

        # Check text labels first
        if (
            self.component1_text is not None
            and self.component1_text.contains(event)[0]
        ):
            self.dragging_component_label = 1
            return
        if (
            self.component2_text is not None
            and self.component2_text.contains(event)[0]
        ):
            self.dragging_component_label = 2
            return

        # Check if we clicked on a component dot
        if (
            self.component1_dot is not None
            and self.component1_dot.contains(event)[0]
        ):
            self.dragging_component = 1
            self.press_event = event
        elif (
            self.component2_dot is not None
            and self.component2_dot.contains(event)[0]
        ):
            self.dragging_component = 2
            self.press_event = event

    def _on_motion(self, event):
        """Handle mouse motion for dragging components or labels."""
        if event.inaxes is None:
            return

        # Dragging a label
        if self.dragging_component_label is not None:
            x, y = event.xdata, event.ydata
            if (
                self.dragging_component_label == 1
                and self.component1_text is not None
            ):
                self.component1_text.set_position((x, y))
                if self.component1_dot is not None:
                    dx, dy = self.component1_dot.get_data()
                    self.component1_text_offset = (x - dx[0], y - dy[0])
            elif (
                self.dragging_component_label == 2
                and self.component2_text is not None
            ):
                self.component2_text.set_position((x, y))
                if self.component2_dot is not None:
                    dx, dy = self.component2_dot.get_data()
                    self.component2_text_offset = (x - dx[0], y - dy[0])
            if self.parent_widget is not None:
                self.parent_widget.canvas_widget.canvas.draw_idle()
            return

        # Dragging a component dot
        if self.dragging_component is None:
            return

        x, y = event.xdata, event.ydata

        if self.dragging_component == 1 and self.component1_dot is not None:
            self.component1_dot.set_data([x], [y])
            self.first_edit1.setText(f"{x:.6f}")
            self.first_edit2.setText(f"{y:.6f}")
            if self.component1_text is not None:
                ox, oy = self.component1_text_offset
                self.component1_text.set_position((x + ox, y + oy))
        elif self.dragging_component == 2 and self.component2_dot is not None:
            self.component2_dot.set_data([x], [y])
            self.second_edit1.setText(f"{x:.6f}")
            self.second_edit2.setText(f"{y:.6f}")
            if self.component2_text is not None:
                ox, oy = self.component2_text_offset
                self.component2_text.set_position((x + ox, y + oy))

        self.draw_line_between_components()

    def _on_release(self, event):
        """Handle mouse release for dragging components."""
        self.dragging_component = None
        self.dragging_component_label = None
        self.press_event = None

    def on_first_button_clicked(self):
        """Function called when the first button is clicked."""
        if self.parent_widget is None:
            show_error("Parent widget not available")
            return

        # Remove existing dot and text if they exist
        if self.component1_dot is not None:
            self.component1_dot.remove()
            self.component1_dot = None
        if self.component1_text is not None:
            self.component1_text.remove()
            self.component1_text = None
        # Remove line as well since component 1 changed
        if self.component_line is not None:
            try:
                self.component_line.remove()
            except (ValueError, AttributeError):
                pass
            self.component_line = None
        self.parent_widget.canvas_widget.canvas.draw_idle()

        original_text = self.my_button.text()
        self.my_button.setText("Click on plot...")
        self.my_button.setEnabled(False)

        # Disconnect original click handler
        if hasattr(self.parent_widget, 'click_cid'):
            self.parent_widget.canvas_widget.canvas.mpl_disconnect(
                self.parent_widget.click_cid
            )

        # Create a new click handler for component 1
        def handle_first_component_click(event):
            if not event.inaxes:
                return

            x, y = event.xdata, event.ydata

            try:
                self.first_edit1.setText(f"{x:.6f}")
                self.first_edit2.setText(f"{y:.6f}")

                component1_color, _ = self._get_component_colors()

                ax = self.parent_widget.canvas_widget.figure.gca()
                self.component1_dot = ax.plot(
                    x,
                    y,
                    'o',
                    color=component1_color,
                    markersize=8,
                    label='Component 1',
                )[0]

                name = self.comp1_name_edit.text().strip()
                if name:
                    ox, oy = self.component1_text_offset
                    self.component1_text = ax.text(
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

            except Exception as e:
                show_error(f"Error setting coordinates: {str(e)}")
            finally:
                # Disconnect temporary handler
                self.parent_widget.canvas_widget.canvas.mpl_disconnect(
                    self.temp_click_cid
                )

                # Restore button state
                self.my_button.setText(original_text)
                self.my_button.setEnabled(True)

        # Connect temporary click handler
        self.temp_click_cid = (
            self.parent_widget.canvas_widget.canvas.mpl_connect(
                'button_press_event', handle_first_component_click
            )
        )

    def on_second_button_clicked(self):
        """Function called when the second button is clicked."""
        if self.parent_widget is None:
            show_error("Parent widget not available")
            return

        # Remove existing dot and text if they exist
        if self.component2_dot is not None:
            self.component2_dot.remove()
            self.component2_dot = None
        if self.component2_text is not None:
            self.component2_text.remove()
            self.component2_text = None
        # Remove line as well since component 2 changed
        if self.component_line is not None:
            try:
                self.component_line.remove()
            except (ValueError, AttributeError):
                pass
            self.component_line = None
        self.parent_widget.canvas_widget.canvas.draw_idle()

        original_text = self.second_button.text()
        self.second_button.setText("Click on plot...")
        self.second_button.setEnabled(False)

        # Disconnect original click handler
        if hasattr(self.parent_widget, 'click_cid'):
            self.parent_widget.canvas_widget.canvas.mpl_disconnect(
                self.parent_widget.click_cid
            )

        # Create a new click handler for component 2
        def handle_second_component_click(event):
            if not event.inaxes:
                return

            x, y = event.xdata, event.ydata

            try:
                self.second_edit1.setText(f"{x:.6f}")
                self.second_edit2.setText(f"{y:.6f}")

                _, component2_color = self._get_component_colors()

                ax = self.parent_widget.canvas_widget.figure.gca()
                self.component2_dot = ax.plot(
                    x,
                    y,
                    'o',
                    color=component2_color,
                    markersize=8,
                    label='Component 2',
                )[0]

                name = self.comp2_name_edit.text().strip()
                if name:
                    ox, oy = self.component2_text_offset
                    self.component2_text = ax.text(
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

            except Exception as e:
                show_error(f"Error setting coordinates: {str(e)}")

            finally:
                # Disconnect temporary handler
                self.parent_widget.canvas_widget.canvas.mpl_disconnect(
                    self.temp_click_cid
                )

                # Restore button state
                self.second_button.setText(original_text)
                self.second_button.setEnabled(True)

        # Connect temporary click handler
        self.temp_click_cid = (
            self.parent_widget.canvas_widget.canvas.mpl_connect(
                'button_press_event', handle_second_component_click
            )
        )

    def on_calculate_button_clicked(self):
        """Function called when the calculate button is clicked."""
        if self.parent_widget._labels_layer_with_phasor_features is None:
            return
        if self.component1_dot is None or self.component2_dot is None:
            return

        component_real = (
            self.component1_dot.get_data()[0][0],
            self.component2_dot.get_data()[0][0],
        )
        component_imag = (
            self.component1_dot.get_data()[1][0],
            self.component2_dot.get_data()[1][0],
        )

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

        fractions_layer_name = f"Component 1 fractions: {self.parent_widget.image_layer_with_phasor_features_combobox.currentText()}"
        selected_fractions_layer = Image(
            fractions,
            name=fractions_layer_name,
            scale=self.parent_widget._labels_layer_with_phasor_features.scale,
            colormap='plasma',
            contrast_limits=(0, 1),
        )

        # Check if the layer is in the viewer before attempting to remove it
        if fractions_layer_name in self.viewer.layers:
            self.viewer.layers.remove(self.viewer.layers[fractions_layer_name])

        self.fractions_layer = self.viewer.add_layer(selected_fractions_layer)
        self.fractions_colormap = self.fractions_layer.colormap.colors
        self.colormap_contrast_limits = self.fractions_layer.contrast_limits

        # Connect to both colormap and contrast limits events
        self.fractions_layer.events.colormap.connect(self._on_colormap_changed)
        self.fractions_layer.events.contrast_limits.connect(
            self._on_contrast_limits_changed
        )

        self.draw_line_between_components()
