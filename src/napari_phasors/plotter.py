import contextlib
import copy
import html
import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from biaplotter.plotter import CanvasWidget
from matplotlib.colorbar import Colorbar
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch
from napari.layers import Image, Labels, Shapes
from napari.utils import colormaps, notifications
from phasorpy.lifetime import phasor_from_lifetime
from qtpy import uic
from qtpy.QtCore import QEvent, Qt, QTimer
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ._utils import (
    CheckableComboBox,
    ColormapLegendHandler,
    ColormapLegendProxy,
    HistogramDockWidget,
    HistogramWidget,
    StatisticsDockWidget,
    apply_filter_and_threshold,
    populate_colormap_combobox,
    resolve_colormap_by_name,
    update_frequency_in_metadata,
)
from .calibration_tab import CalibrationWidget
from .components_tab import ComponentsWidget
from .filter_tab import FilterWidget
from .fret_tab import FretWidget
from .phasor_mapping_tab import PhasorMappingWidget
from .selection_tab import SelectionWidget


class MaskAssignmentDialog(QDialog):
    """Dialog to assign a mask layer to each selected image layer.

    Parameters
    ----------
    image_layer_names : list of str
        Names of the selected image layers.
    mask_layer_names : list of str
        Names of available mask layers (Labels/Shapes).
    current_assignments : dict, optional
        Current mapping of image layer name -> mask layer name.
    parent : QWidget, optional
        Parent widget.
    """

    def __init__(
        self,
        image_layer_names,
        mask_layer_names,
        current_assignments=None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Assign Mask Layers")
        self.setMinimumWidth(400)

        if current_assignments is None:
            current_assignments = {}

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Assign a mask layer to each image layer:"))

        # Scrollable form for assignments
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        form_widget = QWidget()
        form_layout = QFormLayout(form_widget)
        form_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        self._combos = {}  # image_layer_name -> QComboBox
        mask_options = ["None"] + list(mask_layer_names)

        for name in image_layer_names:
            combo = QComboBox()
            combo.addItems(mask_options)
            current = current_assignments.get(name, "None")
            if current in mask_options:
                combo.setCurrentText(current)
            else:
                combo.setCurrentText("None")
            self._combos[name] = combo
            form_layout.addRow(QLabel(name), combo)

        scroll.setWidget(form_widget)
        layout.addWidget(scroll)

        # Apply same mask to all
        apply_all_layout = QHBoxLayout()
        apply_all_layout.addWidget(QLabel("Set all to:"))
        self._apply_all_combo = QComboBox()
        self._apply_all_combo.addItems(mask_options)
        self._apply_all_combo.currentTextChanged.connect(
            self._on_apply_all_changed
        )
        apply_all_layout.addWidget(self._apply_all_combo, 1)
        layout.addLayout(apply_all_layout)

        # OK / Cancel
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _on_apply_all_changed(self, text):
        """Automatically set all per-layer combos when a mask is selected."""
        self._apply_to_all()

    def _apply_to_all(self):
        """Set all combos to the same mask layer."""
        mask = self._apply_all_combo.currentText()
        for combo in self._combos.values():
            combo.setCurrentText(mask)

    def get_assignments(self):
        """Return the mask assignments as a dict.

        Returns
        -------
        dict
            Mapping of image layer name -> mask layer name.
        """
        return {
            name: combo.currentText() for name, combo in self._combos.items()
        }


class _ListWidgetCompatWrapper:
    """Compatibility wrapper to provide combobox-like API for the checkable combobox.

    This allows existing code that calls .currentText() to continue working
    by returning the primary (first) selected layer name.
    """

    def __init__(self, plotter_widget):
        self._plotter = plotter_widget

    def currentText(self):
        """Return the primary selected layer name."""
        return self._plotter.get_primary_layer_name()

    def setCurrentText(self, text):
        """Select a layer by name (clears other selections)."""
        combo = self._plotter.image_layers_checkable_combobox
        combo.setCheckedItems([text])

    @property
    def currentIndexChanged(self):
        """Return the selectionChanged signal for compatibility."""
        return self._plotter.image_layers_checkable_combobox.selectionChanged

    @property
    def currentTextChanged(self):
        """Return the selectionChanged signal for compatibility.

        Note: This is emitted on selection changes, not strictly on text edit.
        """
        return self._plotter.image_layers_checkable_combobox.selectionChanged


class ContourLayerSettingsDialog(QDialog):
    """Dialog for contour multi-layer display and grouping settings."""

    DISPLAY_MODES = ("Merged", "Individual layers", "Grouped")
    COLOR_MODES = ("Colormap", "Solid Color")
    MAX_GROUPS = 10

    def __init__(
        self,
        *,
        display_mode="Merged",
        merged_colormap="turbo",
        merged_style="colormap",
        merged_color=None,
        show_legend=True,
        grouped_color_mode="Use colormap",
        layer_labels=None,
        group_assignments=None,
        layer_colors=None,
        group_colors=None,
        group_names=None,
        layer_styles=None,
        group_styles=None,
        available_colormaps=None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Contour Layer Settings")
        self.setMinimumWidth(550)

        self._layer_labels = list(layer_labels or [])
        self._group_row_data = []
        self._available_colormaps = list(available_colormaps or [])
        if not self._available_colormaps:
            self._available_colormaps = list(colormaps.ALL_COLORMAPS.keys())

        group_assignments = dict(group_assignments or {})
        layer_colors = dict(layer_colors or {})
        group_colors = dict(group_colors or {})
        group_names = dict(group_names or {})
        layer_styles = dict(layer_styles or {})
        group_styles = dict(group_styles or {})

        root = QVBoxLayout(self)

        # Display mode
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Multi-layer display mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(list(self.DISPLAY_MODES))
        self.mode_combo.setCurrentText(display_mode)
        mode_layout.addWidget(self.mode_combo)
        root.addLayout(mode_layout)

        default_tab10 = plt.cm.tab10.colors

        # Merged mode style
        self._merged_colormap_layout = QHBoxLayout()
        self._merged_colormap_label = QLabel("Merged contour colormap:")
        self._merged_colormap_combo = QComboBox()
        self._populate_colormap_combobox(
            self._merged_colormap_combo, selected=merged_colormap
        )
        if merged_style == "solid":
            self._merged_colormap_combo.setCurrentIndex(0)
        self._merged_colormap_combo.currentTextChanged.connect(
            self._on_merged_colormap_changed
        )
        self._merged_color_btn = QPushButton()
        self._merged_color_btn.setFixedSize(24, 24)
        if merged_color is None:
            merged_color = default_tab10[0]
        self._set_btn_color(self._merged_color_btn, merged_color)
        self._merged_color_btn.clicked.connect(
            lambda checked=False, b=self._merged_color_btn: self._pick_color(b)
        )
        self._merged_colormap_layout.addWidget(self._merged_colormap_label)
        self._merged_colormap_layout.addWidget(self._merged_colormap_combo)
        self._merged_colormap_layout.addWidget(self._merged_color_btn)
        root.addLayout(self._merged_colormap_layout)

        self._show_legend_checkbox = QCheckBox("Show legend")
        self._show_legend_checkbox.setChecked(bool(show_legend))
        root.addWidget(self._show_legend_checkbox)

        # Track the current style mode (colormap or solid)
        self._current_merged_style = merged_style

        # Per-layer colors section
        self._layer_section = QWidget()
        layer_section_layout = QVBoxLayout(self._layer_section)
        layer_section_layout.setContentsMargins(0, 0, 0, 0)
        layer_section_layout.addWidget(QLabel("Individual layer styles:"))
        self._layer_color_buttons = {}
        self._layer_style_widgets = {}
        for i, label in enumerate(self._layer_labels):
            row = QHBoxLayout()
            row.addWidget(QLabel(label))

            cmap_combo = QComboBox()
            self._populate_colormap_combobox(
                cmap_combo, selected=merged_colormap
            )
            cmap_combo.currentTextChanged.connect(
                lambda text, layer_name=label, combo=cmap_combo, btn=None: self._on_layer_colormap_changed(
                    layer_name, text, combo
                )
            )
            row.addWidget(cmap_combo)

            btn = QPushButton()
            btn.setFixedSize(24, 24)
            color = layer_colors.get(
                label, default_tab10[i % len(default_tab10)]
            )
            self._set_btn_color(btn, color)
            btn.clicked.connect(
                lambda checked=False, b=btn: self._pick_color(b)
            )
            row.addWidget(btn)
            row.addStretch(1)
            layer_section_layout.addLayout(row)

            row_style = dict(layer_styles.get(label, {}))
            row_cmap = row_style.get("colormap", merged_colormap)
            if row_cmap in self._available_colormaps:
                cmap_combo.setCurrentText(row_cmap)
            elif row_style.get("mode") == "solid":
                cmap_combo.setCurrentIndex(0)

            self._layer_color_buttons[label] = btn
            self._layer_style_widgets[label] = {
                "cmap_combo": cmap_combo,
                "color_btn": btn,
            }

        root.addWidget(self._layer_section)

        # Group section
        self._group_section = QWidget()
        group_layout = QVBoxLayout(self._group_section)
        group_layout.setContentsMargins(0, 0, 0, 0)
        group_layout.addWidget(QLabel("Grouped styles:"))

        self._group_rows_widget = QWidget()
        self._group_rows_layout = QVBoxLayout(self._group_rows_widget)
        self._group_rows_layout.setContentsMargins(0, 0, 0, 0)
        self._group_rows_layout.setSpacing(4)
        group_layout.addWidget(self._group_rows_widget)

        if group_assignments and self._layer_labels:
            grouped = {}
            for name in self._layer_labels:
                gid = int(group_assignments.get(name, 1))
                grouped.setdefault(gid, []).append(name)
            for gid in sorted(grouped):
                self._add_group_row(
                    name=group_names.get(gid, f"Group {gid}"),
                    color=group_colors.get(
                        gid, default_tab10[(gid - 1) % len(default_tab10)]
                    ),
                    checked_layers=grouped[gid],
                    style=group_styles.get(gid, {}).get(
                        "mode",
                        (
                            "solid"
                            if grouped_color_mode == "Use custom colors"
                            else "colormap"
                        ),
                    ),
                    colormap_name=group_styles.get(gid, {}).get(
                        "colormap", merged_colormap
                    ),
                )
        elif self._layer_labels:
            self._add_group_row(
                name="Group 1",
                checked_layers=list(self._layer_labels),
                style=(
                    "solid"
                    if grouped_color_mode == "Use custom colors"
                    else "colormap"
                ),
                colormap_name=merged_colormap,
            )

        add_group_btn = QPushButton("+ Add Group")
        add_group_btn.setMaximumWidth(120)
        add_group_btn.clicked.connect(self._on_add_group)
        group_layout.addWidget(add_group_btn)
        root.addWidget(self._group_section)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self.mode_combo.currentTextChanged.connect(self._update_ui_for_mode)
        self._update_ui_for_mode(self.mode_combo.currentText())

    @staticmethod
    def _set_btn_color(btn, color):
        r, g, b = color[:3]
        btn._color = (float(r), float(g), float(b))
        btn.setStyleSheet(
            f"background-color: rgb({int(r*255)}, {int(g*255)}, {int(b*255)});"
        )

    def _pick_color(self, btn):
        from qtpy.QtWidgets import QColorDialog

        r, g, b = btn._color
        initial = QColorDialog().currentColor()
        initial.setRgbF(r, g, b)
        chosen = QColorDialog.getColor(initial, self)
        if chosen.isValid():
            self._set_btn_color(btn, chosen.getRgbF()[:3])

    def _on_merged_colormap_changed(self, text):
        """Handle colormap selection in merged mode, including 'Select color...'."""
        if text == "Select color...":
            self._pick_color(self._merged_color_btn)
            self._current_merged_style = "solid"
            self._merged_colormap_combo.blockSignals(True)
            self._merged_colormap_combo.setCurrentIndex(0)
            self._merged_colormap_combo.blockSignals(False)
        else:
            self._current_merged_style = "colormap"
        self._update_custom_color_visibility()

    def _on_layer_colormap_changed(self, layer_name, text, combo):
        """Handle colormap selection for individual layers, including 'Select color...'."""
        if text == "Select color...":
            btn = self._layer_color_buttons.get(layer_name)
            if btn:
                self._pick_color(btn)
            combo.blockSignals(True)
            combo.setCurrentIndex(0)
            combo.blockSignals(False)
        self._update_custom_color_visibility()

    def _on_group_colormap_changed(self, group_idx, text, combo):
        """Handle colormap selection for groups, including 'Select color...'."""
        if text == "Select color...":
            if group_idx < len(self._group_row_data):
                btn = self._group_row_data[group_idx]["color_btn"]
                self._pick_color(btn)
            combo.blockSignals(True)
            combo.setCurrentIndex(0)
            combo.blockSignals(False)
        self._update_custom_color_visibility()

    def _update_ui_for_mode(self, mode):
        is_individual = mode == "Individual layers"
        is_grouped = mode == "Grouped"
        is_merged = mode == "Merged"
        self._merged_colormap_label.setVisible(is_merged)
        self._merged_colormap_combo.setVisible(is_merged)
        self._layer_section.setVisible(is_individual)
        self._group_section.setVisible(is_grouped)
        self._update_custom_color_visibility()

    def _update_custom_color_visibility(self):
        mode = self.mode_combo.currentText()
        is_individual = mode == "Individual layers"
        is_grouped = mode == "Grouped"
        is_merged = mode == "Merged"

        self._merged_color_btn.setVisible(
            is_merged
            and self._merged_colormap_combo.currentText() == "Select color..."
        )

        for layer_name in self._layer_labels:
            self._update_layer_style_row(layer_name, visible=is_individual)

        for row_data in self._group_row_data:
            cmap_combo = row_data["cmap_combo"]
            color_btn = row_data["color_btn"]
            cmap_combo.setVisible(is_grouped)
            color_btn.setVisible(
                is_grouped and cmap_combo.currentText() == "Select color..."
            )

    def _update_layer_style_row(self, layer_name, visible=True):
        row = self._layer_style_widgets.get(layer_name)
        if row is None:
            return
        row["cmap_combo"].setVisible(visible)
        row["color_btn"].setVisible(
            visible and row["cmap_combo"].currentText() == "Select color..."
        )

    def _populate_colormap_combobox(self, combo, selected=None):
        populate_colormap_combobox(
            combo,
            include_select_color=True,
            selected=selected,
            available_colormaps=self._available_colormaps,
        )

    def _add_group_row(
        self,
        name="Group",
        color=None,
        checked_layers=None,
        style="colormap",
        colormap_name="turbo",
    ):
        default_tab10 = plt.cm.tab10.colors
        idx = len(self._group_row_data)
        if color is None:
            color = default_tab10[idx % len(default_tab10)]

        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(4)

        name_edit = QLineEdit(name)
        name_edit.setMaximumWidth(120)
        row_layout.addWidget(name_edit)

        cmap_combo = QComboBox()
        self._populate_colormap_combobox(cmap_combo, selected=colormap_name)
        if style == "solid":
            cmap_combo.setCurrentIndex(0)
        cmap_combo.currentTextChanged.connect(
            lambda text, idx=idx, combo=cmap_combo: self._on_group_colormap_changed(
                idx, text, combo
            )
        )
        row_layout.addWidget(cmap_combo)

        color_btn = QPushButton()
        color_btn.setFixedSize(24, 24)
        self._set_btn_color(color_btn, color)
        color_btn.clicked.connect(
            lambda checked=False, b=color_btn: self._pick_color(b)
        )
        row_layout.addWidget(color_btn)

        layer_combo = CheckableComboBox(
            placeholder="Select layers...",
            parent=self,
            enable_primary_layer=False,
        )
        layer_combo.addItems(self._layer_labels)
        if checked_layers:
            layer_combo.setCheckedItems(checked_layers)
        else:
            # Explicitly call _update_display_text to show placeholder
            layer_combo._update_display_text()
        row_layout.addWidget(layer_combo, 1)

        remove_btn = QPushButton("-")
        remove_btn.setFixedSize(24, 24)
        remove_btn.setToolTip("Remove this group")
        remove_btn.clicked.connect(lambda: self._on_remove_group(row_widget))
        row_layout.addWidget(remove_btn)

        self._group_rows_layout.addWidget(row_widget)
        self._group_row_data.append(
            {
                "container": row_widget,
                "name_edit": name_edit,
                "cmap_combo": cmap_combo,
                "color_btn": color_btn,
                "layer_combo": layer_combo,
            }
        )
        self._update_custom_color_visibility()

    def _on_add_group(self):
        if len(self._group_row_data) >= self.MAX_GROUPS:
            return
        self._add_group_row(name=f"Group {len(self._group_row_data) + 1}")

    def _on_remove_group(self, row_widget):
        if len(self._group_row_data) <= 1:
            return
        idx = None
        for i, data in enumerate(self._group_row_data):
            if data["container"] is row_widget:
                idx = i
                break
        if idx is None:
            return
        row = self._group_row_data.pop(idx)
        row["container"].setParent(None)

    def get_display_mode(self):
        return self.mode_combo.currentText()

    def get_individual_color_mode(self):
        # Always use colormap mode (style selector removed)
        return "Use colormap"

    def get_grouped_color_mode(self):
        # Always use colormap mode (style selector removed)
        return "Use colormap"

    def get_merged_colormap(self):
        return self._merged_colormap_combo.currentText()

    def get_merged_style(self):
        return self._current_merged_style

    def get_merged_color(self):
        return self._merged_color_btn._color

    def get_show_legend(self):
        return self._show_legend_checkbox.isChecked()

    def get_layer_styles(self):
        styles = {}
        for name, row in self._layer_style_widgets.items():
            selected = row["cmap_combo"].currentText()
            is_solid = selected == "Select color..."
            styles[name] = {
                "mode": "solid" if is_solid else "colormap",
                "colormap": selected,
                "color": row["color_btn"]._color,
            }
        return styles

    def get_group_styles(self):
        styles = {}
        for gid, row in enumerate(self._group_row_data, start=1):
            selected = row["cmap_combo"].currentText()
            is_solid = selected == "Select color..."
            styles[gid] = {
                "mode": "solid" if is_solid else "colormap",
                "colormap": selected,
                "color": row["color_btn"]._color,
            }
        return styles

    def get_layer_colors(self):
        return {
            name: style["color"]
            for name, style in self.get_layer_styles().items()
        }

    def get_group_assignments(self):
        assignments = {}
        for gid, row in enumerate(self._group_row_data, start=1):
            for layer_name in row["layer_combo"].checkedItems():
                assignments[layer_name] = gid
        return assignments

    def get_group_names(self):
        names = {}
        for gid, row in enumerate(self._group_row_data, start=1):
            text = row["name_edit"].text().strip()
            names[gid] = text if text else f"Group {gid}"
        return names

    def get_group_colors(self):
        return {
            gid: style["color"]
            for gid, style in self.get_group_styles().items()
        }


class PlotterWidget(QWidget):
    """A widget for plotting phasor features.

    This widget contains a fixed canvas widget at the top for plotting phasor features
    and a tabbed interface below with different analysis tools.

    Parameters
    ----------
    napari_viewer : napari.Viewer
        The napari viewer object.

    Attributes
    ----------
    viewer : napari.Viewer
        The napari viewer object.
    canvas_widget : biaplotter.plotter.CanvasWidget
        The canvas widget for plotting phasor features (fixed at the top).
    image_layer_with_phasor_features_listwidget : QListWidget
        The list widget for selecting multiple image layers with phasor features.
    image_layers_checkable_combobox : CheckableComboBox
        The dropdown combobox for selecting multiple image layers with phasor features.
        Supports multi-selection via checkboxes for merged plotting.
    harmonic_spinbox : QSpinBox
        The spinbox for selecting the harmonic.
    tab_widget : QTabWidget
        The tab widget containing different analysis tools.
    settings_tab : QWidget
        The Settings tab containing the main plotting controls.
    selection_tab : QWidget
        The selection tab for cursor-based analysis.
    components_tab : QWidget
        The Components tab for component analysis.
    phasor_mapping_tab : QWidget
        The Phasor Mapping tab for phasor mapping analysis.
    fret_tab : QWidget
        The FRET tab for FRET analysis.
    plotter_inputs_widget : QWidget
        The main plotter inputs widget (in Settings tab). The widget contains:
        - semi_circle_checkbox : QCheckBox
            The checkbox for toggling the display of the semi-circle plot.
        - white_background_checkbox : QCheckBox
            The checkbox for toggling the white background in the plot.
        - plot_type_combobox : QComboBox
            The combobox for selecting the plot type.
        - colormap_combobox : QComboBox
            The combobox for selecting the histogram colormap.
        - number_of_bins_spinbox : QSpinBox
            The spinbox for selecting the number of bins in the histogram.
        - log_scale_checkbox : QCheckBox
            The checkbox for selecting the log scale in the histogram.

    """

    def __init__(self, napari_viewer):
        """Initialize the PlotterWidget."""
        super().__init__()
        self._is_closing = False
        self.viewer = napari_viewer

        self.setWindowTitle("Phasor Plot")
        self.setObjectName("Phasor Plot")

        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().setSpacing(2)

        # Initialize data attributes
        self._g_array = None
        self._s_array = None
        self._g_original_array = None
        self._s_original_array = None
        self._harmonics_array = None

        # Cache for histogram properties to avoid redundant updates
        self._last_histogram_bins = None
        self._last_histogram_colormap = None
        self._last_histogram_colormap_object = (
            None  # Store the actual colormap object
        )
        self._last_histogram_norm = None
        self._last_histogram_color_indices = None
        self._last_scatter_color_indices = None

        # User-defined axes limits: set when user explicitly zooms/pans.
        # When not None, these limits are always restored after every plot()
        # so that layer selection changes never reset the user's zoom.
        # Cleared only on fresh layer load.
        self._user_axes_limits = None

        # Create top widget for canvas
        self.canvas_container = QWidget()
        self.canvas_container.setLayout(QVBoxLayout())
        self.canvas_container.layout().setContentsMargins(0, 0, 0, 0)
        self.canvas_container.layout().setSpacing(0)
        self.canvas_container.layout().setAlignment(Qt.AlignCenter)
        self.canvas_container.installEventFilter(self)
        self.layout().addWidget(
            self.canvas_container, 1
        )  # stretch factor 1 to prioritize canvas

        # Load canvas widget (fixed at the top)
        self.canvas_widget = CanvasWidget(
            napari_viewer, highlight_enabled=False
        )
        self.canvas_widget.axes.set_aspect(1, adjustable='box')
        self.canvas_widget.setMinimumSize(0, 0)
        self.canvas_widget.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        self.set_axes_labels()
        self.canvas_container.layout().addWidget(self.canvas_widget)

        # Monkey-patch biaplotter's _is_click_inside_axes to handle None xdata/ydata
        # This fixes a bug where clicking outside the axes causes a TypeError
        original_is_click_inside = self.canvas_widget._is_click_inside_axes

        def _is_click_inside_axes_fixed(event):
            if event.xdata is None or event.ydata is None:
                return False
            return original_is_click_inside(event)

        self.canvas_widget._is_click_inside_axes = _is_click_inside_axes_fixed

        # Monkey-patch toolbar save_figure to export with black text/spines
        self._patch_toolbar_save()

        # Patch toolbar zoom/pan release to capture user-set limits
        self._patch_toolbar_limits()

        # Connect scroll wheel zoom on the phasor plot
        self.canvas_widget.canvas.mpl_connect(
            "scroll_event", self._on_scroll_zoom
        )

        # Create bottom widget for controls
        self.controls_container = QWidget()
        self.controls_container.setLayout(QVBoxLayout())
        self.controls_container.layout().setContentsMargins(10, 10, 10, 10)
        self.controls_container.layout().setSpacing(3)
        self.controls_container.setSizePolicy(
            QSizePolicy.Preferred, QSizePolicy.Maximum
        )
        self.controls_container.installEventFilter(self)
        self.layout().addWidget(
            self.controls_container, 0
        )  # stretch factor 0 to keep fixed size

        # Add checkable combobox for multi-layer selection
        image_layer_layout = QHBoxLayout()
        image_layer_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins
        image_layer_layout.setSpacing(5)  # Reduce spacing between widgets
        image_layer_layout.addWidget(QLabel("Image Layers:"))
        self.image_layers_checkable_combobox = CheckableComboBox()
        self.image_layers_checkable_combobox.setToolTip(
            "Select one or more layers to plot. Check multiple layers to merge their phasor data.\n"
            "Click 'Set as primary' next to a layer name to change the primary layer.\n"
            "The primary layer is used for settings and analysis."
        )
        image_layer_layout.addWidget(self.image_layers_checkable_combobox, 1)

        # "All | None" clickable labels for quick bulk selection
        select_all_label = QLabel('<a href="all" style="color: gray;">All</a>')
        select_all_label.setTextFormat(Qt.RichText)
        select_all_label.setCursor(Qt.PointingHandCursor)
        select_all_label.setToolTip("Select all layers")
        image_layer_layout.addWidget(select_all_label)

        separator_label = QLabel("|")
        separator_label.setStyleSheet("color: gray;")
        image_layer_layout.addWidget(separator_label)

        deselect_all_label = QLabel(
            '<a href="none" style="color: gray;">None</a>'
        )
        deselect_all_label.setTextFormat(Qt.RichText)
        deselect_all_label.setCursor(Qt.PointingHandCursor)
        deselect_all_label.setToolTip("Deselect all layers")
        image_layer_layout.addWidget(deselect_all_label)

        # Connect All/None labels (use lambdas to consume the href argument)
        select_all_label.linkActivated.connect(
            lambda _: self.image_layers_checkable_combobox.selectAll()
        )
        deselect_all_label.linkActivated.connect(
            lambda _: self.image_layers_checkable_combobox.deselectAll()
        )

        image_layer_widget = QWidget()
        image_layer_widget.setLayout(image_layer_layout)
        self.controls_container.layout().addWidget(image_layer_widget)

        # Create a horizontal box for harmonic and mask controls
        harmonics_and_mask_container = QHBoxLayout()
        harmonics_and_mask_container.setContentsMargins(
            0, 0, 0, 0
        )  # Remove margins
        harmonics_and_mask_container.setSpacing(
            5
        )  # Reduce spacing between widgets
        # Harmonic label and spinbox (left side)
        self.harmonic_label = QLabel("Harmonic:")
        harmonics_and_mask_container.addWidget(self.harmonic_label)
        self.harmonic_spinbox = QSpinBox()
        self.harmonic_spinbox.setMinimum(1)
        self.harmonic_spinbox.setValue(1)
        harmonics_and_mask_container.addWidget(self.harmonic_spinbox, 1)

        # Per-layer mask assignments: {image_layer_name: mask_layer_name}
        self._mask_assignments = {}

        # Mask label and combobox (shown when 0-1 layers selected)
        self.mask_layer_label = QLabel("Mask Layer:")
        harmonics_and_mask_container.addWidget(self.mask_layer_label)
        self.mask_layer_combobox = QComboBox()
        self.mask_layer_combobox.setToolTip(
            "Create or select a Labels or Shapes layer with a mask to restrict analysis to specific regions. "
            "Selecting 'None' will disable masking."
        )
        self.mask_layer_combobox.addItem("None")
        harmonics_and_mask_container.addWidget(self.mask_layer_combobox, 1)

        # Mask assign button (shown when >1 layers selected)
        self.mask_assign_button = QPushButton("Assign Masks...")
        self.mask_assign_button.setToolTip(
            "Assign different mask layers to each selected image layer."
        )
        self.mask_assign_button.clicked.connect(
            self._open_mask_assignment_dialog
        )
        self.mask_assign_button.setVisible(False)
        harmonics_and_mask_container.addWidget(self.mask_assign_button, 1)

        self.controls_container.layout().addLayout(
            harmonics_and_mask_container
        )

        # Add import buttons below harmonic spinbox
        import_buttons_layout = QHBoxLayout()
        import_buttons_layout.setContentsMargins(0, 0, 0, 0)
        import_buttons_layout.setSpacing(5)

        import_label = QLabel("Load and Apply Settings from:")
        import_buttons_layout.addWidget(import_label)

        self.import_from_layer_button = QPushButton("Layer")
        import_buttons_layout.addWidget(self.import_from_layer_button)

        self.import_from_file_button = QPushButton("OME-TIFF File")
        import_buttons_layout.addWidget(self.import_from_file_button)

        import_buttons_widget = QWidget()
        import_buttons_widget.setLayout(import_buttons_layout)
        self.controls_container.layout().addWidget(import_buttons_widget)

        # Dynamic buttons to re-open closed dock widgets
        dock_buttons_layout = QHBoxLayout()
        dock_buttons_layout.setContentsMargins(0, 0, 0, 0)
        dock_buttons_layout.setSpacing(5)

        self.show_analysis_button = QPushButton("Show Analysis Tabs")
        self.show_analysis_button.setToolTip(
            "Re-open the Phasor Analysis dock"
        )
        self.show_analysis_button.setVisible(False)
        self.show_analysis_button.clicked.connect(self._show_analysis_dock)
        dock_buttons_layout.addWidget(self.show_analysis_button)

        self.show_histogram_button = QPushButton("Show Histogram")
        self.show_histogram_button.setToolTip("Re-open the Histogram dock")
        self.show_histogram_button.setVisible(False)
        self.show_histogram_button.clicked.connect(self._show_histogram_dock)
        dock_buttons_layout.addWidget(self.show_histogram_button)

        self.show_statistics_button = QPushButton("Show Statistics Table")
        self.show_statistics_button.setToolTip("Re-open the Statistics dock")
        self.show_statistics_button.setVisible(False)
        self.show_statistics_button.clicked.connect(self._show_statistics_dock)
        dock_buttons_layout.addWidget(self.show_statistics_button)

        self._dock_buttons_widget = QWidget()
        self._dock_buttons_widget.setLayout(dock_buttons_layout)
        self._dock_buttons_widget.setVisible(False)
        self.controls_container.layout().addWidget(self._dock_buttons_widget)

        # Timer that polls dock visibility — reliable for both the hide and
        # close-X buttons (the latter destroys the dock without emitting signals).
        self._dock_check_timer = QTimer(self)
        self._dock_check_timer.timeout.connect(self._check_dock_visibility)
        self._dock_check_timer.start(500)

        # Create tab widget
        self.tab_widget = QTabWidget()

        # Create a separate widget for the tabs to allow independent docking
        self.analysis_widget = QWidget()
        self.analysis_widget.setLayout(QVBoxLayout())
        self.analysis_widget.layout().addWidget(self.tab_widget)

        # Create a shared histogram container using a QStackedWidget.
        # Page 0 is an empty placeholder; pages 1-3 hold
        # Lifetime / FRET / Components histogram dock widgets.
        self._histogram_stack = QStackedWidget()
        # Empty placeholder page: a HistogramWidget with no data (buttons disabled)
        self._histogram_empty_widget = HistogramWidget()
        self._histogram_stack.addWidget(
            self._histogram_empty_widget
        )  # index 0
        self._histogram_stack.setCurrentIndex(0)

        # Statistics stacked widget (mirrors the histogram stack structure)
        self._statistics_stack = QStackedWidget()
        # Empty placeholder page: a StatisticsDockWidget driven by a dummy HistogramWidget
        self._stats_empty_hist_widget = HistogramWidget()
        self._statistics_empty_widget = StatisticsDockWidget(
            self._stats_empty_hist_widget
        )
        self._statistics_stack.addWidget(
            self._statistics_empty_widget
        )  # index 0
        self._statistics_stack.setCurrentIndex(0)

        # Wrapper for the statistics dock
        self.statistics_container = QWidget()
        self.statistics_container.setLayout(QVBoxLayout())
        self.statistics_container.layout().setContentsMargins(0, 0, 0, 0)
        self._statistics_title_label = QLabel("Statistics")
        self._statistics_title_label.setStyleSheet(
            "font-weight: bold; font-size: 13px; padding: 4px 0px;"
        )
        self._statistics_title_label.setAlignment(Qt.AlignCenter)
        self.statistics_container.layout().addWidget(
            self._statistics_title_label
        )
        self.statistics_container.layout().addWidget(self._statistics_stack)
        self.statistics_container.setMinimumWidth(300)

        # Wrapper so the dock widget gets a nice title
        self.histogram_container = QWidget()
        self.histogram_container.setLayout(QVBoxLayout())
        self.histogram_container.layout().setContentsMargins(0, 0, 0, 0)
        # Dynamic title label that updates based on the active tab
        self._histogram_title_label = QLabel("Histogram")
        self._histogram_title_label.setStyleSheet(
            "font-weight: bold; font-size: 13px; padding: 4px 0px;"
        )
        self._histogram_title_label.setAlignment(Qt.AlignCenter)
        self.histogram_container.layout().addWidget(
            self._histogram_title_label
        )
        self.histogram_container.layout().addWidget(self._histogram_stack)
        # Prevent the histogram from being shrunk below a usable size
        self.histogram_container.setMinimumWidth(350)

        # Add the analysis widget to the viewer with a delay to ensure
        # correct dock ordering. Use owned timers so teardown can cancel them.
        self._analysis_dock_init_timer = QTimer(self)
        self._analysis_dock_init_timer.setSingleShot(True)
        self._analysis_dock_init_timer.timeout.connect(
            self._add_analysis_dock_widget
        )

        self._dock_resize_timer = QTimer(self)
        self._dock_resize_timer.setSingleShot(True)
        self._dock_resize_timer.timeout.connect(self._resize_initial_docks)

        self._analysis_dock_init_timer.start(20)

        # Keep controls at their natural height and expand the canvas to the
        # largest square that fits in the remaining space.
        self._resize_canvas_to_available_space()

        # Add a flag to prevent recursive calls
        self._updating_plot = False
        self._updating_settings = False

        # Debounce timers for expensive operations
        self._layer_selection_timer = QTimer(self)
        self._layer_selection_timer.setSingleShot(True)
        self._layer_selection_timer.setInterval(300)  # 300ms delay
        self._layer_selection_timer.timeout.connect(
            self._process_layer_selection_change
        )

        self._bins_timer = QTimer(self)
        self._bins_timer.setSingleShot(True)
        self._bins_timer.setInterval(500)  # 500ms delay
        self._bins_timer.timeout.connect(self._process_bins_change)

        # Create Settings tab
        self.settings_tab = QWidget()
        self.settings_tab.setLayout(QVBoxLayout())
        self.tab_widget.addTab(self.settings_tab, "Plot Settings")

        # Load plotter inputs widget from ui file
        self.plotter_inputs_widget = QWidget()
        uic.loadUi(
            Path(__file__).parent / "ui/plotter_inputs_widget.ui",
            self.plotter_inputs_widget,
        )
        self.settings_tab.layout().addWidget(self.plotter_inputs_widget)
        self.setMinimumSize(300, 400)

        # Create other tabs
        self._create_calibration_tab()
        self._create_filter_tab()
        self._create_selection_tab()
        self._create_components_tab()
        self._create_phasor_mapping_tab()
        self._create_fret_tab()

        # Connect napari signals when new layer is inseted or removed
        self.viewer.layers.events.inserted.connect(self.reset_layer_choices)
        self.viewer.layers.events.removed.connect(self.reset_layer_choices)

        # Set contour single-layer defaults before wiring UI signals.
        # Combo-box population can emit callbacks during initialization.
        self._histogram_style = "colormap"
        self._histogram_colormap_name = "turbo"
        self._histogram_color = (0.1216, 0.4667, 0.7059)
        self._single_contour_style = "colormap"
        self._single_contour_colormap = "turbo"
        self._single_contour_color = (0.1216, 0.4667, 0.7059)
        self._preserve_plot_type_on_restore = False

        # Connect callbacks
        # When primary layer changes, update all tab UIs (but don't run analyses)
        self.image_layers_checkable_combobox.primaryLayerChanged.connect(
            self._on_primary_layer_changed
        )
        # When selection changes, only update the plot
        self.image_layers_checkable_combobox.selectionChanged.connect(
            self._on_selection_changed
        )
        # Update mask UI mode when selection changes (combobox vs button)
        self.image_layers_checkable_combobox.selectionChanged.connect(
            self._update_mask_ui_mode
        )
        # Update all frequency widgets from layer metadata if primary layer changes
        self.image_layers_checkable_combobox.primaryLayerChanged.connect(
            self._sync_frequency_inputs_from_metadata
        )
        # Update mask when mask layer selection changes
        self.mask_layer_combobox.currentTextChanged.connect(
            self._on_mask_layer_changed
        )
        self.plotter_inputs_widget.semi_circle_checkbox.stateChanged.connect(
            self._on_semi_circle_changed
        )
        self.harmonic_spinbox.valueChanged.connect(self._on_harmonic_changed)
        self.plotter_inputs_widget.plot_type_combobox.currentIndexChanged.connect(
            self._on_plot_type_changed
        )
        self.plotter_inputs_widget.colormap_combobox.currentIndexChanged.connect(
            self._on_colormap_changed
        )
        self.plotter_inputs_widget.number_of_bins_spinbox.valueChanged.connect(
            self._on_bins_changed
        )
        self.plotter_inputs_widget.log_scale_checkbox.stateChanged.connect(
            self._on_log_scale_changed
        )
        self.plotter_inputs_widget.white_background_checkbox.stateChanged.connect(
            self._on_white_background_changed
        )

        self.plotter_inputs_widget.marker_size_spinbox.valueChanged.connect(
            self._on_marker_size_changed
        )
        self.plotter_inputs_widget.marker_color_button.clicked.connect(
            self._on_marker_color_clicked
        )
        self.plotter_inputs_widget.marker_alpha_spinbox.valueChanged.connect(
            self._on_marker_alpha_changed
        )
        self.plotter_inputs_widget.contour_levels_spinbox.valueChanged.connect(
            self._on_contour_levels_changed
        )
        self.plotter_inputs_widget.contour_linewidth_spinbox.valueChanged.connect(
            self._on_contour_linewidth_changed
        )

        contour_settings_parent = self.plotter_inputs_widget.findChild(
            QWidget, "scrollAreaWidgetContents"
        )
        contour_settings_layout = contour_settings_parent.layout()

        self.plotter_inputs_widget.colormap_combobox.setParent(None)
        self._colormap_row_widget = QWidget()
        self._colormap_row_layout = QHBoxLayout(self._colormap_row_widget)
        self._colormap_row_layout.setContentsMargins(0, 0, 0, 0)
        self._colormap_row_layout.setSpacing(4)
        self._colormap_row_layout.addWidget(
            self.plotter_inputs_widget.colormap_combobox, 1
        )
        self.plotter_inputs_widget.contour_single_color_button = QPushButton()
        combo_h = (
            self.plotter_inputs_widget.colormap_combobox.sizeHint().height()
        )
        self.plotter_inputs_widget.contour_single_color_button.setFixedSize(
            combo_h, combo_h
        )
        self.plotter_inputs_widget.contour_single_color_button.setToolTip(
            "Choose solid contour color"
        )
        self.plotter_inputs_widget.contour_single_color_button.clicked.connect(
            self._on_single_contour_color_clicked
        )
        self.plotter_inputs_widget.contour_single_color_button.setVisible(
            False
        )
        self._colormap_row_layout.addWidget(
            self.plotter_inputs_widget.contour_single_color_button
        )
        self._colormap_row_widget.setMinimumHeight(combo_h)
        self._colormap_row_widget.setMaximumHeight(combo_h)
        contour_settings_layout.addWidget(self._colormap_row_widget, 3, 1)

        contour_row = 3
        widgets_to_shift = []
        for i in range(contour_settings_layout.count()):
            item = contour_settings_layout.itemAt(i)
            widget = item.widget()
            if widget is None:
                continue
            row, col, row_span, col_span = (
                contour_settings_layout.getItemPosition(i)
            )
            if row >= contour_row:
                widgets_to_shift.append((widget, row, col, row_span, col_span))

        # Shift bottom-up to avoid overlap while re-inserting.
        for widget, row, col, row_span, col_span in sorted(
            widgets_to_shift, key=lambda x: x[1], reverse=True
        ):
            contour_settings_layout.removeWidget(widget)
            contour_settings_layout.addWidget(
                widget,
                row + 1,
                col,
                row_span,
                col_span,
            )

        self.plotter_inputs_widget.label_contour_layer_settings = QLabel(
            "Multi-layer contour display:"
        )
        self.plotter_inputs_widget.contour_layer_settings_button = QPushButton(
            "Configure Multi-Layer Display..."
        )
        self.plotter_inputs_widget.contour_layer_settings_button.clicked.connect(
            self._on_contour_layer_settings_clicked
        )
        contour_settings_layout.addWidget(
            self.plotter_inputs_widget.label_contour_layer_settings,
            contour_row,
            0,
        )
        contour_settings_layout.addWidget(
            self.plotter_inputs_widget.contour_layer_settings_button,
            contour_row,
            1,
        )
        self._plotter_settings_layout = contour_settings_layout

        self.import_from_layer_button.clicked.connect(
            self._import_settings_from_layer
        )
        self.import_from_file_button.clicked.connect(
            self._import_settings_from_file
        )

        # Populate plot type combobox
        self.plotter_inputs_widget.plot_type_combobox.addItems(
            ['HISTOGRAM2D', 'SCATTER', 'CONTOUR']
        )

        # Populate colormap combobox with swatches.
        self._populate_main_colormap_combobox(
            selected=self._single_contour_colormap,
        )
        self.histogram_colormap = "turbo"

        # Initialize attributes
        self.polar_plot_artist_list = []
        self.semi_circle_plot_artist_list = []
        self._contour_collections = []
        self._contour_display_mode = "Merged"
        self._contour_layer_colors = {}
        self._contour_group_assignments = {}
        self._contour_group_colors = {}
        self._contour_group_names = {}
        self._contour_multi_layer_colormap = "turbo"
        self._contour_merged_style = "colormap"
        self._contour_merged_color = (0.1216, 0.4667, 0.7059)
        self._contour_layer_styles = {}
        self._contour_group_styles = {}
        self._contour_show_legend = False
        self.toggle_semi_circle = True
        self.colorbar = None
        self._colormap = self.canvas_widget.artists[
            'HISTOGRAM2D'
        ].overlay_colormap
        self._histogram_colormap = self.canvas_widget.artists[
            'HISTOGRAM2D'
        ].histogram_colormap

        # Start with the histogram2d plot type
        self.plot_type = 'HISTOGRAM2D'
        self._current_plot_type = 'HISTOGRAM2D'

        # Initialize the dynamic UI elements
        self._on_plot_type_changed()
        self._update_contour_controls_visibility()
        self._update_scatter_colormap()

        # Connect only the initial active artist
        self._connect_active_artist_signals()
        self._connect_selector_signals()
        self.canvas_widget.show_color_overlay_signal.connect(
            self._enforce_axes_aspect
        )

        # Set the initial plot
        self._redefine_axes_limits()
        self._update_plot_bg_color()

        # Populate labels layer combobox
        self.reset_layer_choices()

        # Connect tab change signal
        self.tab_widget.currentChanged.connect(self._on_tab_changed)
        self._set_selection_visibility(False)

    def _add_analysis_dock_widget(self):
        """Add the analysis widget and histogram container to the viewer.

        Histogram and statistics use separate dock widgets that are tabified
        together in the same bottom area, alongside the analysis dock.
        """
        if (
            hasattr(self.viewer, 'window')
            and self.viewer.window is not None
            and hasattr(self.viewer.window, '_qt_window')
        ):
            # Create histogram and analysis first, then split them.
            # Create statistics after that and tabify only with histogram.
            self._histogram_dock = self.viewer.window.add_dock_widget(
                self.histogram_container,
                name="Histogram",
                area="bottom",
            )

            # Analysis dock — rightmost
            self._analysis_dock = self.viewer.window.add_dock_widget(
                self.analysis_widget,
                name="Phasor Analysis",
                area="bottom",
            )

            self._statistics_dock = self.viewer.window.add_dock_widget(
                self.statistics_container,
                name="Statistics",
                area="bottom",
            )
            self._docks_initialized = True

            self._enforce_bottom_dock_layout()

            # Defer resizeDocks so it runs after Qt has applied the splits.
            self._dock_resize_timer.start(200)

    def _resize_initial_docks(self):
        """Resize docks after delayed split has been applied."""
        if not getattr(self, '_docks_initialized', False):
            return
        with contextlib.suppress(AttributeError, RuntimeError):
            qt_window = self.viewer.window._qt_window
            qt_window.resizeDocks(
                [
                    self._histogram_dock,
                    self._analysis_dock,
                ],
                [600, 500],
                Qt.Horizontal,
            )

    def _tabify_histogram_and_statistics_docks(self):
        """Tabify the histogram and statistics docks in the bottom area."""
        if not hasattr(self, '_histogram_dock') or not hasattr(
            self, '_statistics_dock'
        ):
            return
        with contextlib.suppress(AttributeError, RuntimeError):
            qt_window = self.viewer.window._qt_window
            qt_window.tabifyDockWidget(
                self._histogram_dock,
                self._statistics_dock,
            )
            self._histogram_dock.raise_()

    def _enforce_bottom_dock_layout(self):
        """Keep analysis separate and tabify only histogram/statistics."""
        if not all(
            hasattr(self, name)
            for name in (
                '_histogram_dock',
                '_statistics_dock',
                '_analysis_dock',
            )
        ):
            return

        with contextlib.suppress(AttributeError, RuntimeError):
            qt_window = self.viewer.window._qt_window
            qt_window.splitDockWidget(
                self._histogram_dock,
                self._analysis_dock,
                Qt.Horizontal,
            )
        self._tabify_histogram_and_statistics_docks()

    def eventFilter(self, obj, event):
        """Recompute square canvas size when relevant containers resize."""
        watched = {
            getattr(self, 'canvas_container', None),
            getattr(self, 'controls_container', None),
        }
        if obj in watched and event.type() in {QEvent.Resize, QEvent.Show}:
            QTimer.singleShot(0, self._resize_canvas_to_available_space)
        return super().eventFilter(obj, event)

    def resizeEvent(self, event):
        """Keep the phasor canvas square while maximizing available area."""
        super().resizeEvent(event)
        self._resize_canvas_to_available_space()

    def _resize_canvas_to_available_space(self):
        """Resize canvas to the largest square fitting the top container."""
        if not hasattr(self, 'canvas_container') or not hasattr(
            self, 'canvas_widget'
        ):
            return
        available_w = self.canvas_container.width()
        available_h = self.canvas_container.height()
        side = max(min(available_w, available_h), 1)
        self.canvas_widget.setFixedSize(side, side)

    def get_selected_layer_names(self):
        """Get the names of all selected (checked) layers.

        Returns
        -------
        list of str
            List of checked layer names.
        """
        return self.image_layers_checkable_combobox.checkedItems()

    def get_primary_layer_name(self):
        """Get the name of the primary selected layer.

        The primary layer is used for metadata operations and analysis.
        This is determined by the CheckableComboBox's primary layer tracking.

        Returns
        -------
        str
            Name of the primary selected layer, or empty string if none.
        """
        return self.image_layers_checkable_combobox.getPrimaryLayer()

    def get_selected_layers(self):
        """Get all selected layer objects.

        Returns
        -------
        list of napari.layers.Image
            List of selected Image layer objects.
        """
        selected_names = self.get_selected_layer_names()
        return [
            self.viewer.layers[name]
            for name in selected_names
            if name in self.viewer.layers
        ]

    def get_primary_layer(self):
        """Get the primary (first) selected layer object.

        Returns
        -------
        napari.layers.Image or None
            The primary selected layer, or None if none selected.
        """
        primary_name = self.get_primary_layer_name()
        if primary_name and primary_name in self.viewer.layers:
            return self.viewer.layers[primary_name]
        return None

    # Backward compatibility property
    @property
    def image_layer_with_phasor_features_combobox(self):
        """Backward compatibility property.

        Returns a wrapper object that provides currentText() for
        compatibility with code that still uses the old combobox API.
        """
        return _ListWidgetCompatWrapper(self)

    def _get_default_plot_settings(self):
        """Get default settings dictionary for plot parameters."""

        default_harmonic = 1
        layer_name = (
            self.image_layer_with_phasor_features_combobox.currentText()
        )
        if layer_name:
            try:
                layer_metadata = self.viewer.layers[layer_name].metadata
                if "harmonics" in layer_metadata:
                    harmonics = layer_metadata["harmonics"]
                    if harmonics is not None:
                        default_harmonic = int(np.min(harmonics))
            except (KeyError, AttributeError, ValueError):
                pass

        return {
            'harmonic': default_harmonic,
            'semi_circle': self.toggle_semi_circle,
            'white_background': self.white_background,
            'plot_type': self.plot_type,
            'colormap': self.histogram_colormap,
            'histogram_style': self._histogram_style,
            'histogram_color': self._histogram_color,
            'number_of_bins': self.histogram_bins,
            'log_scale': self.histogram_log_scale,
            'marker_size': 50,
            'marker_alpha': 0.5,
            'marker_color': '#1f77b4',
            'contour_levels': 7,
            'contour_linewidth': 1.5,
            'contour_display_mode': "Merged",
            'contour_layer_colors': {},
            'contour_group_assignments': {},
            'contour_group_colors': {},
            'contour_group_names': {},
            'contour_multi_layer_colormap': self.histogram_colormap,
            'contour_merged_style': 'colormap',
            'contour_merged_color': (0.1216, 0.4667, 0.7059),
            'contour_layer_styles': {},
            'contour_group_styles': {},
            'contour_show_legend': False,
            'contour_single_style': 'colormap',
            'contour_single_colormap': self.histogram_colormap,
            'contour_single_color': (0.1216, 0.4667, 0.7059),
        }

    def _initialize_plot_settings_in_metadata(self, layer):
        """Initialize settings in layer metadata if not present."""
        if 'settings' not in layer.metadata:
            layer.metadata['settings'] = {}

        default_settings = self._get_default_plot_settings()
        for key, default_value in default_settings.items():
            if key not in layer.metadata['settings']:
                layer.metadata['settings'][key] = default_value

    def _update_setting_in_metadata(self, key, value):
        """Update a specific setting in the current layer's metadata."""
        if self._updating_settings:
            return

        layer_name = (
            self.image_layer_with_phasor_features_combobox.currentText()
        )
        if layer_name:
            layer = self.viewer.layers[layer_name]
            if 'settings' not in layer.metadata:
                layer.metadata['settings'] = {}
            layer.metadata['settings'][key] = value

    def _restore_plot_settings_from_metadata(self):
        """Restore all settings from the current layer's metadata."""
        layer_name = (
            self.image_layer_with_phasor_features_combobox.currentText()
        )
        if not layer_name:
            return

        image_layer = self.viewer.layers[layer_name]
        if 'settings' not in image_layer.metadata:
            self._initialize_plot_settings_in_metadata(image_layer)
            return

        self._updating_settings = True
        try:
            if 'mask' in image_layer.metadata:
                # Find a mask layer that matches the saved mask
                matching_mask_layer_name = None
                valid_mask_layers = [
                    mask_l
                    for mask_l in self.viewer.layers
                    if isinstance(mask_l, (Labels, Shapes))
                ]
                for mask_l in valid_mask_layers:
                    if isinstance(mask_l, Shapes):
                        mask_data = mask_l.to_labels(
                            labels_shape=image_layer.data.shape
                        )
                    else:
                        mask_data = mask_l.data
                    if np.array_equal(mask_data, image_layer.metadata['mask']):
                        matching_mask_layer_name = mask_l.name
                        break  # Found a match, no need to continue
                # Create mask layer if no match found
                if matching_mask_layer_name is None:
                    matching_mask_layer_name = f"Restored Mask: {layer_name}"
                    self.viewer.add_labels(
                        image_layer.metadata['mask'],
                        name=matching_mask_layer_name,
                        scale=image_layer.scale,
                    )
                self.mask_layer_combobox.setCurrentText(
                    matching_mask_layer_name
                )

            settings = image_layer.metadata['settings']

            # Restore harmonic
            if 'harmonic' in settings:
                self.harmonic_spinbox.setValue(settings['harmonic'])

            # Only restore if explicitly set in metadata
            if 'white_background' in settings:
                self.plotter_inputs_widget.white_background_checkbox.setChecked(
                    settings['white_background']
                )
                # Update axes labels and plot background immediately
                self.set_axes_labels()
                self._update_plot_bg_color()

            # Only restore if explicitly set in metadata
            if 'semi_circle' in settings:
                # Use the setter to properly update the display
                self.toggle_semi_circle = settings['semi_circle']

            # Only restore if explicitly set in metadata
            if (
                'plot_type' in settings
                and not self._preserve_plot_type_on_restore
            ):
                self.plotter_inputs_widget.plot_type_combobox.setCurrentText(
                    settings['plot_type']
                )

            # Only restore if explicitly set in metadata
            if 'colormap' in settings:
                self.plotter_inputs_widget.colormap_combobox.setCurrentText(
                    settings['colormap']
                )

            # Only restore if explicitly set in metadata
            if 'number_of_bins' in settings:
                self.plotter_inputs_widget.number_of_bins_spinbox.setValue(
                    settings['number_of_bins']
                )

            # Only restore if explicitly set in metadata
            if 'log_scale' in settings:
                self.plotter_inputs_widget.log_scale_checkbox.setChecked(
                    settings['log_scale']
                )

            if 'marker_size' in settings:
                self.plotter_inputs_widget.marker_size_spinbox.setValue(
                    settings['marker_size']
                )

            if 'marker_alpha' in settings:
                self.plotter_inputs_widget.marker_alpha_spinbox.setValue(
                    settings['marker_alpha']
                )

            if 'contour_levels' in settings:
                self.plotter_inputs_widget.contour_levels_spinbox.setValue(
                    settings['contour_levels']
                )

            if 'contour_linewidth' in settings:
                self.plotter_inputs_widget.contour_linewidth_spinbox.setValue(
                    settings['contour_linewidth']
                )

            self._contour_display_mode = settings.get(
                'contour_display_mode', "Merged"
            )
            self._contour_layer_colors = settings.get(
                'contour_layer_colors', {}
            )
            self._contour_group_assignments = settings.get(
                'contour_group_assignments', {}
            )
            self._contour_group_colors = settings.get(
                'contour_group_colors', {}
            )
            self._contour_group_names = settings.get('contour_group_names', {})
            self._contour_multi_layer_colormap = settings.get(
                'contour_multi_layer_colormap',
                settings.get('colormap', self.histogram_colormap),
            )
            self._contour_merged_style = settings.get(
                'contour_merged_style', 'colormap'
            )
            self._contour_merged_color = tuple(
                settings.get('contour_merged_color', (0.1216, 0.4667, 0.7059))
            )
            self._contour_layer_styles = settings.get(
                'contour_layer_styles',
                {},
            )
            self._contour_group_styles = settings.get(
                'contour_group_styles',
                {},
            )
            self._contour_show_legend = bool(
                settings.get('contour_show_legend', False)
            )
            self._single_contour_style = settings.get(
                'contour_single_style', 'colormap'
            )
            self._single_contour_colormap = settings.get(
                'contour_single_colormap', settings.get('colormap', 'turbo')
            )
            self._single_contour_color = tuple(
                settings.get('contour_single_color', (0.1216, 0.4667, 0.7059))
            )
            self._histogram_colormap_name = settings.get('colormap', 'turbo')
            self._histogram_style = settings.get('histogram_style', 'colormap')
            self._histogram_color = tuple(
                settings.get('histogram_color', (0.1216, 0.4667, 0.7059))
            )
            self._update_single_contour_color_button()

            # Backward compatibility: convert legacy color settings to styles.
            if not self._contour_layer_styles:
                default_mode = 'colormap'
                self._contour_layer_styles = {
                    name: {
                        'mode': default_mode,
                        'colormap': self._contour_multi_layer_colormap,
                        'color': tuple(color),
                    }
                    for name, color in (
                        self._contour_layer_colors or {}
                    ).items()
                }

            if not self._contour_group_styles:
                default_mode = 'colormap'
                self._contour_group_styles = {
                    int(gid): {
                        'mode': default_mode,
                        'colormap': self._contour_multi_layer_colormap,
                        'color': tuple(color),
                    }
                    for gid, color in (
                        self._contour_group_colors or {}
                    ).items()
                }

            if 'marker_color' in settings:
                color = settings['marker_color']
                self._marker_color = color
                self.plotter_inputs_widget.marker_color_button.setStyleSheet(
                    f"background-color: {color};"
                )
                self._update_scatter_colormap()

            self._refresh_main_colormap_control_for_mode()

        finally:
            self._updating_settings = False

    def _show_import_dialog(self, default_checked=None, source_settings=None):
        """Show a dialog to select which analyses to import/apply.

        Parameters
        ----------
        default_checked : list, optional
            List of tab keys that should be checked by default
        source_settings : dict, optional
            Settings dictionary from source layer/file to determine which tabs to show
        """
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Analyses to Import")
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("Select which analyses to import and apply:"))

        # Frequency checkbox (always show if frequency exists in settings)
        frequency_cb = None
        if source_settings is not None and "frequency" in source_settings:
            frequency_cb = QCheckBox("Frequency")
            frequency_cb.setChecked(
                default_checked is None
                or "frequency"
                in (
                    default_checked
                    if isinstance(default_checked, list)
                    else []
                )
            )
            layout.addWidget(frequency_cb)

            line = QFrame()
            line.setFrameShape(QFrame.HLine)
            line.setFrameShadow(QFrame.Sunken)
            layout.addWidget(line)

        # Map tab display names to their metadata keys
        tab_mapping = [
            ("Plot Settings", "settings_tab", None),
            ("Calibration", "calibration_tab", "calibrated"),
            (
                "Filter",
                "filter_tab",
                ["threshold", "filter"],
            ),
            (
                "Phasor Mapping",
                "phasor_mapping_tab",
                ["phasor_mapping", "lifetime"],
            ),
            ("FRET", "fret_tab", "fret"),
            ("Components", "components_tab", "component_analysis"),
            ("Selection", "selection_tab", "selections"),
        ]

        checkboxes = {}

        for label, attr, settings_key in tab_mapping:
            # Determine if this tab should be shown
            show_tab = False
            if source_settings is None or settings_key is None:
                show_tab = True
            elif isinstance(settings_key, list):
                show_tab = any(key in source_settings for key in settings_key)
            else:
                show_tab = settings_key in source_settings

            if show_tab:
                cb = QCheckBox(label)
                cb.setChecked(
                    default_checked is None
                    or attr
                    in (
                        default_checked
                        if isinstance(default_checked, list)
                        else []
                    )
                )
                layout.addWidget(cb)
                checkboxes[attr] = cb

        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        layout.addWidget(button_box)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)

        dialog.setLayout(layout)
        if dialog.exec_() == QDialog.Accepted:
            selected = [
                attr for attr, cb in checkboxes.items() if cb.isChecked()
            ]
            if frequency_cb is not None and frequency_cb.isChecked():
                selected.insert(0, "frequency")
            return selected
        return []

    def _restore_all_tab_analyses(self, selected_tabs=None):
        """Restore UI values for selected tabs from metadata.

        This only restores UI state (input fields, sliders, visual elements
        on the plot) but does NOT run any analyses. The user must click
        the respective run/apply/calculate buttons to execute analyses.
        """
        if selected_tabs is None:
            selected_tabs = [
                "settings_tab",
                "calibration_tab",
                "filter_tab",
                "phasor_mapping_tab",
                "fret_tab",
                "components_tab",
                "selection_tab",
            ]

        # Backward compatibility: legacy selections may still use "lifetime_tab".
        if (
            "lifetime_tab" in selected_tabs
            and "phasor_mapping_tab" not in selected_tabs
        ):
            selected_tabs = [*selected_tabs, "phasor_mapping_tab"]

        # Restore plot settings first if selected
        if "settings_tab" in selected_tabs:
            self._restore_plot_settings_from_metadata()

        if "calibration_tab" in selected_tabs and hasattr(
            self, 'calibration_tab'
        ):
            self.calibration_tab._on_image_layer_changed()
        if "filter_tab" in selected_tabs and hasattr(self, 'filter_tab'):
            self.filter_tab._on_image_layer_changed()
        if "components_tab" in selected_tabs and hasattr(
            self, 'components_tab'
        ):
            self.components_tab._on_image_layer_changed()
        if "phasor_mapping_tab" in selected_tabs and hasattr(
            self, 'phasor_mapping_tab'
        ):
            self.phasor_mapping_tab._on_image_layer_changed()
        if "fret_tab" in selected_tabs and hasattr(self, 'fret_tab'):
            self.fret_tab._on_image_layer_changed()
        if "selection_tab" in selected_tabs and hasattr(self, 'selection_tab'):
            self.selection_tab._on_image_layer_changed()

        current_tab_index = self.tab_widget.currentIndex()
        self._on_tab_changed(current_tab_index)

    def _apply_calibration_if_needed(self):
        """Apply calibration transformation if needed to all selected layers."""
        selected_layers = self.get_selected_layers()
        if not selected_layers:
            return
        for layer in selected_layers:
            settings = layer.metadata.get("settings", {})
            if (
                settings.get("calibrated", False)
                and "calibration_phase" in settings
                and "calibration_modulation" in settings
                and not layer.metadata.get("calibration_applied", False)
            ):
                phi_zero = settings["calibration_phase"]
                mod_zero = settings["calibration_modulation"]
                self.calibration_tab._apply_phasor_transformation(
                    layer.name, phi_zero, mod_zero
                )
                layer.metadata["calibration_applied"] = True

    def _get_import_settings_groups(self):
        """Return the settings keys controlled by each importable tab."""
        return {
            "settings_tab": [
                "harmonic",
                "semi_circle",
                "white_background",
                "plot_type",
                "colormap",
                "histogram_style",
                "histogram_color",
                "number_of_bins",
                "log_scale",
                "marker_size",
                "marker_alpha",
                "marker_color",
                "contour_levels",
                "contour_linewidth",
                "contour_display_mode",
                "contour_layer_colors",
                "contour_group_assignments",
                "contour_group_colors",
                "contour_group_names",
                "contour_multi_layer_colormap",
                "contour_merged_style",
                "contour_merged_color",
                "contour_layer_styles",
                "contour_group_styles",
                "contour_show_legend",
                "contour_single_style",
                "contour_single_colormap",
                "contour_single_color",
            ],
            "calibration_tab": [
                "calibrated",
                "calibration_phase",
                "calibration_modulation",
            ],
            "filter_tab": [
                "filter",
                "threshold",
                "threshold_upper",
                "threshold_method",
            ],
            "phasor_mapping_tab": ["phasor_mapping", "lifetime"],
            "fret_tab": ["fret"],
            "components_tab": ["component_analysis"],
            "selection_tab": ["selections"],
        }

    def _merge_imported_settings(
        self, current_settings, source_settings, selected_tabs
    ):
        """Merge only the settings groups selected for import."""
        merged_settings = copy.deepcopy(current_settings or {})
        source_settings = source_settings or {}

        for tab_name, keys in self._get_import_settings_groups().items():
            if tab_name not in selected_tabs:
                continue

            for key in keys:
                merged_settings.pop(key, None)

            for key in keys:
                if key in source_settings:
                    merged_settings[key] = copy.deepcopy(source_settings[key])

        return merged_settings

    def _prepare_layer_for_import(self, layer, selected_tabs):
        """Prepare the target layer before applying imported analyses."""
        settings = layer.metadata.get("settings", {})

        if "calibration_tab" in selected_tabs and settings.get(
            "calibrated", False
        ):
            self.calibration_tab._uncalibrate_layer(layer.name)

        if "calibration_tab" in selected_tabs:
            layer.metadata.pop("calibration_applied", None)

    def _apply_imported_filter_if_needed(self, layer):
        """Apply imported filter and threshold settings to a layer."""
        settings = layer.metadata.get("settings", {})
        filter_settings = settings.get("filter") or {}
        filter_method = filter_settings.get("method")
        harmonics = (
            layer.metadata.get("harmonics")
            if filter_method == "wavelet"
            else None
        )

        apply_filter_and_threshold(
            layer,
            threshold=settings.get("threshold"),
            threshold_upper=settings.get("threshold_upper"),
            threshold_method=settings.get("threshold_method"),
            filter_method=filter_method,
            size=filter_settings.get("size"),
            repeat=filter_settings.get("repeat"),
            sigma=filter_settings.get("sigma"),
            levels=filter_settings.get("levels"),
            harmonics=harmonics,
        )

    def _apply_imported_analyses(self, layers, selected_tabs):
        """Apply imported analysis settings that affect layer data.

        Parameters
        ----------
        layers : list of napari.layers.Image or napari.layers.Image
            The target layer(s) to apply analyses to.
        selected_tabs : list of str
            The tabs whose analyses should be applied.
        """
        # Support both single layer and list of layers for backward compat
        if not isinstance(layers, list):
            layers = [layers]

        if "calibration_tab" in selected_tabs:
            self._apply_calibration_if_needed()

        if "filter_tab" in selected_tabs:
            for layer in layers:
                self._apply_imported_filter_if_needed(layer)
            self.refresh_phasor_data()

    def _import_settings_from_layer(self):
        """Import all settings and analyses from another layer, with selection dialog."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Import Settings from Layer")
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("Select layer to import settings from:"))

        layer_combo = QComboBox()
        selected_layer_names = set(self.get_selected_layer_names())
        available_layers = [
            layer.name
            for layer in self.viewer.layers
            if isinstance(layer, Image)
            and all(
                key in layer.metadata
                for key in ["G", "S", "G_original", "S_original"]
            )
            and layer.name not in selected_layer_names
        ]
        layer_combo.addItems(available_layers)
        layout.addWidget(layer_combo)

        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        layout.addWidget(button_box)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        dialog.setLayout(layout)

        if dialog.exec_() == QDialog.Accepted and layer_combo.currentText():
            source_layer_name = layer_combo.currentText()
            source_layer = self.viewer.layers[source_layer_name]

            source_settings = source_layer.metadata.get('settings', {}).copy()

            selected_tabs = self._show_import_dialog(
                source_settings=source_settings
            )
            if selected_tabs:
                self._copy_metadata_from_layer(
                    source_layer_name, selected_tabs
                )

    def _import_settings_from_file(self):
        """Import settings from an OME-TIFF file, with selection dialog."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select OME-TIFF file",
            "",
            "OME-TIFF Files (*.ome.tif *.ome.tiff);;All Files (*)",
        )
        if not file_path:
            return
        try:
            import json

            from phasorpy import io

            _, _, _, attrs = io.phasor_from_ometiff(file_path, harmonic='all')
            settings = {}
            if "frequency" in attrs:
                settings["frequency"] = attrs["frequency"]
            if "description" in attrs:
                try:
                    # HTML-unescape the description to handle tifffile HTML encoding
                    description_str = html.unescape(attrs["description"])
                    description = json.loads(description_str)
                    if "napari_phasors_settings" in description:
                        napari_phasors_settings = json.loads(
                            description["napari_phasors_settings"]
                        )
                        for key, value in napari_phasors_settings.items():
                            settings[key] = value
                except (json.JSONDecodeError, KeyError):
                    notifications.WarningNotification(
                        "Failed to parse napari-phasors settings from file"
                    )
            if settings:
                selected_tabs = self._show_import_dialog(
                    source_settings=settings
                )
                if selected_tabs:
                    self._apply_imported_settings(settings, selected_tabs)
                    notifications.show_info(
                        f"Settings imported from {Path(file_path).name}"
                    )
            else:
                notifications.WarningNotification(
                    "No valid napari-phasors settings found in file"
                )
        except Exception as e:  # noqa: BLE001
            notifications.WarningNotification(
                f"Failed to import from file: {str(e)}"
            )

    def _copy_metadata_from_layer(self, source_layer_name, selected_tabs):
        """Copy metadata from source layer to all selected layers and apply selected analyses."""
        try:
            source_layer = self.viewer.layers[source_layer_name]
            selected_layers = self.get_selected_layers()
            if not selected_layers:
                notifications.WarningNotification("No layer selected")
                return

            source_settings = copy.deepcopy(
                source_layer.metadata.get('settings', {})
            )

            selected_analysis_tabs = [
                tab for tab in selected_tabs if tab != "frequency"
            ]

            for target_layer in selected_layers:
                current_settings = copy.deepcopy(
                    target_layer.metadata.get('settings', {})
                )

                self._prepare_layer_for_import(
                    target_layer, selected_analysis_tabs
                )

                target_layer.metadata['settings'] = (
                    self._merge_imported_settings(
                        current_settings,
                        source_settings,
                        selected_analysis_tabs,
                    )
                )

                if (
                    "frequency" in selected_tabs
                    and 'frequency' in source_settings
                ):
                    freq_val = source_settings['frequency']
                    update_frequency_in_metadata(target_layer, freq_val)

            if "frequency" in selected_tabs and 'frequency' in source_settings:
                self._broadcast_frequency_value_across_tabs(
                    str(source_settings['frequency'])
                )

            self._apply_imported_analyses(
                selected_layers, selected_analysis_tabs
            )

            self._restore_plot_settings_from_metadata()
            self._restore_all_tab_analyses(selected_tabs)
            self.plot()
            layer_names = ", ".join([layer.name for layer in selected_layers])
            notifications.show_info(
                f"Settings and analyses imported from {source_layer_name} to {layer_names}"
            )
        except Exception as e:  # noqa: BLE001
            notifications.WarningNotification(
                f"Failed to import settings: {str(e)}"
            )

    def _apply_imported_settings(self, settings, selected_tabs):
        selected_layers = self.get_selected_layers()
        if not selected_layers:
            notifications.WarningNotification("No layer selected")
            return

        selected_analysis_tabs = [
            tab for tab in selected_tabs if tab != "frequency"
        ]

        for target_layer in selected_layers:
            current_settings = copy.deepcopy(
                target_layer.metadata.get('settings', {})
            )

            self._prepare_layer_for_import(
                target_layer, selected_analysis_tabs
            )

            target_layer.metadata['settings'] = self._merge_imported_settings(
                current_settings,
                settings,
                selected_analysis_tabs,
            )

            if 'frequency' in selected_tabs and 'frequency' in settings:
                update_frequency_in_metadata(
                    target_layer, settings['frequency']
                )

        if 'frequency' in selected_tabs and 'frequency' in settings:
            self._broadcast_frequency_value_across_tabs(
                str(settings['frequency'])
            )

        self._apply_imported_analyses(selected_layers, selected_analysis_tabs)

        self._restore_plot_settings_from_metadata()
        self._restore_all_tab_analyses(selected_tabs)
        self.plot()

    def _on_tab_changed(self, index):
        """Handle tab change events to show/hide tab-specific lines."""
        current_tab = self.tab_widget.widget(index)

        if hasattr(self, 'phasor_mapping_tab'):
            self.phasor_mapping_tab.on_tab_visibility_changed(
                current_tab == self.phasor_mapping_tab
            )

        self._hide_all_tab_artists()

        self._show_tab_artists(current_tab)

        # Show/hide histogram dock widgets based on active tab
        self._update_histogram_dock_visibility(current_tab)

        # Update filter histogram if switching to filter tab and it needs updating
        if hasattr(self, 'filter_tab') and current_tab == self.filter_tab:
            self.filter_tab._update_histogram_if_needed()

        self.canvas_widget.figure.canvas.draw_idle()

    def _hide_all_tab_artists(self):
        """Hide all tab-specific artists."""
        if hasattr(self, 'selection_tab'):
            # Deactivate any active selection tools before hiding toolbar
            if hasattr(self, 'canvas_widget'):
                self.canvas_widget._on_escape(None)
            self._set_selection_visibility(False)
            self._set_circular_cursor_visibility(False)
        if hasattr(self, 'components_tab'):
            self._set_components_visibility(False)
        if hasattr(self, 'fret_tab'):
            self._set_fret_visibility(False)

    def _clear_all_tab_artists(self):
        """Clear (remove) all tab-specific artists."""
        if hasattr(self, 'selection_tab'):
            # Deactivate any active selection tools before clearing
            if hasattr(self, 'canvas_widget'):
                self.canvas_widget._on_escape(None)
            self.selection_tab.clear_artists()
        if hasattr(self, 'components_tab'):
            self.components_tab.clear_artists()
        if hasattr(self, 'fret_tab'):
            self.fret_tab.clear_artists()

    def _show_tab_artists(self, current_tab):
        """Show artists for the specified tab."""
        if current_tab == getattr(self, 'selection_tab', None):
            # Only show toolbar if manual selection mode is active
            if hasattr(self.selection_tab, 'is_manual_selection_mode'):
                is_manual = self.selection_tab.is_manual_selection_mode()
                self._set_selection_visibility(is_manual)
                if not is_manual:
                    self._set_circular_cursor_visibility(True)
            else:
                self._set_selection_visibility(True)
        elif current_tab == getattr(self, 'components_tab', None):
            self._set_components_visibility(True)
        elif current_tab == getattr(self, 'fret_tab', None):
            self._set_fret_visibility(True)

    def _set_selection_visibility(self, visible):
        """Set visibility of selection toolbar."""
        if hasattr(self, 'selection_tab'):
            layout = self.canvas_widget.selection_tools_layout
            for i in range(layout.count()):
                item = layout.itemAt(i)
                widget = item.widget()
                if widget is not None:
                    widget.setVisible(visible)

    def _set_circular_cursor_visibility(self, visible):
        """Set visibility of circular cursor patches."""
        if hasattr(self, 'selection_tab'):
            circular_cursor_widget = self.selection_tab.circular_cursor_widget
            if visible:
                circular_cursor_widget.redraw_all_patches()
            else:
                circular_cursor_widget.clear_all_patches()

    def _set_components_visibility(self, visible):
        """Set visibility of components tab artists."""
        if hasattr(self, 'components_tab'):
            self.components_tab.set_artists_visible(visible)

    def _set_fret_visibility(self, visible):
        """Set visibility of FRET tab artists."""
        if hasattr(self, 'fret_tab'):
            self.fret_tab.set_artists_visible(visible)

    def _update_histogram_dock_visibility(self, current_tab):
        """Switch the shared histogram and statistics stacks to the active tab.

        For tabs without a histogram (Plot Settings, Calibration, Filter,
        Selection) the empty placeholder page is shown with an informative
        message.
        """
        if current_tab == getattr(self, 'phasor_mapping_tab', None):
            hist_idx = getattr(self, '_phasor_map_hist_page_idx', 0)
            stats_idx = getattr(self, '_phasor_map_stats_page_idx', 0)
            output_display = "Lifetime"
            with contextlib.suppress(AttributeError, RuntimeError):
                output_display = (
                    self.phasor_mapping_tab.get_selected_output_display_name()
                )
            hist_title = f"{output_display} Histogram"
            stats_title = f"{output_display} Statistics"
        elif current_tab == getattr(self, 'fret_tab', None):
            hist_idx = getattr(self, '_fret_hist_page_idx', 0)
            stats_idx = getattr(self, '_fret_stats_page_idx', 0)
            hist_title = "FRET Histogram"
            stats_title = "FRET Statistics"
        elif current_tab == getattr(self, 'components_tab', None):
            hist_idx = getattr(self, '_components_hist_page_idx', 0)
            stats_idx = getattr(self, '_components_stats_page_idx', 0)
            hist_title = "Components Histogram"
            stats_title = "Components Statistics"
        else:
            hist_idx = 0
            stats_idx = 0
            hist_title = "Histogram"
            stats_title = "Statistics"
        self._histogram_stack.setCurrentIndex(hist_idx)
        self._statistics_stack.setCurrentIndex(stats_idx)
        if hasattr(self, '_histogram_title_label'):
            self._histogram_title_label.setText(hist_title)
        if hasattr(self, '_statistics_title_label'):
            self._statistics_title_label.setText(stats_title)

    def _check_dock_visibility(self):
        """Poll dock state and show/hide the re-open buttons accordingly.

        Runs every 500 ms via a QTimer so it correctly detects both the
        'hide' button (which emits visibilityChanged) and the 'close X'
        button (which destroys the dock without emitting any signal).
        """
        if not getattr(self, '_docks_initialized', False):
            return

        def _is_hidden(attr):
            try:
                return not getattr(self, attr).isVisible()
            except (AttributeError, RuntimeError):
                return True

        analysis_hidden = _is_hidden('_analysis_dock')
        histogram_hidden = _is_hidden('_histogram_dock')
        statistics_hidden = _is_hidden('_statistics_dock')

        self.show_analysis_button.setVisible(analysis_hidden)
        self.show_histogram_button.setVisible(histogram_hidden)
        self.show_statistics_button.setVisible(statistics_hidden)
        self._dock_buttons_widget.setVisible(
            analysis_hidden or histogram_hidden or statistics_hidden
        )

    def _show_statistics_dock(self):
        """Make the statistics dock visible, re-adding it if closed."""
        if not hasattr(self, '_statistics_dock'):
            return
        try:
            self._statistics_dock.setVisible(True)
        except RuntimeError:
            self._statistics_dock = self.viewer.window.add_dock_widget(
                self.statistics_container,
                name="Statistics",
                area="bottom",
            )
        self._enforce_bottom_dock_layout()
        with contextlib.suppress(AttributeError, RuntimeError):
            self._statistics_dock.raise_()

    def _show_analysis_dock(self):
        """Make the analysis dock widget visible, re-adding it if it was closed."""
        if not hasattr(self, '_analysis_dock'):
            return
        try:
            self._analysis_dock.setVisible(True)
        except RuntimeError:
            self._analysis_dock = self.viewer.window.add_dock_widget(
                self.analysis_widget,
                name="Phasor Analysis",
                area="bottom",
            )
        self._enforce_bottom_dock_layout()
        with contextlib.suppress(AttributeError, RuntimeError):
            self._analysis_dock.raise_()

    def _show_histogram_dock(self):
        """Make the histogram dock visible, re-adding it if closed."""
        if not hasattr(self, '_histogram_dock'):
            return
        try:
            self._histogram_dock.setVisible(True)
        except RuntimeError:
            self._histogram_dock = self.viewer.window.add_dock_widget(
                self.histogram_container,
                name="Histogram",
                area="bottom",
            )
        self._enforce_bottom_dock_layout()
        with contextlib.suppress(AttributeError, RuntimeError):
            self._histogram_dock.raise_()

    def _on_semi_circle_changed(self, state):
        """Callback for semi circle checkbox change."""
        self._update_setting_in_metadata('semi_circle', bool(state))
        if not self._updating_settings:
            # Clear user zoom so _redefine_axes_limits computes correct
            # default limits for the new mode and updates the toolbar home.
            self._user_axes_limits = None
            self.toggle_semi_circle = bool(state)

    def _on_harmonic_changed(self, value):
        """Callback for harmonic spinbox change."""
        self._update_setting_in_metadata('harmonic', value)
        if not self._updating_settings:
            self.refresh_current_plot()
            self._update_plot_elements()
        if hasattr(self, 'selection_tab'):
            self.selection_tab.on_harmonic_changed()

    def _on_plot_type_changed(self):
        """Callback for plot type change."""
        new_plot_type = (
            self.plotter_inputs_widget.plot_type_combobox.currentText()
        )
        self._update_setting_in_metadata('plot_type', new_plot_type)

        is_scatter = new_plot_type == 'SCATTER'
        is_contour = new_plot_type == 'CONTOUR'

        # Histogram/Contour shared elements
        self.plotter_inputs_widget.label_3.setVisible(not is_scatter)
        self._colormap_row_widget.setVisible(not is_scatter)
        self.plotter_inputs_widget.label_4.setVisible(not is_scatter)
        self.plotter_inputs_widget.number_of_bins_spinbox.setVisible(
            not is_scatter
        )
        show_log_controls = (not is_scatter) and (not is_contour)
        self.plotter_inputs_widget.label_7.setVisible(show_log_controls)
        self.plotter_inputs_widget.log_scale_checkbox.setVisible(
            show_log_controls
        )

        # Scatter plot elements
        self.plotter_inputs_widget.label_marker_size.setVisible(is_scatter)
        self.plotter_inputs_widget.marker_size_spinbox.setVisible(is_scatter)
        self.plotter_inputs_widget.label_marker_color.setVisible(is_scatter)
        self.plotter_inputs_widget.marker_color_button.setVisible(is_scatter)
        self.plotter_inputs_widget.label_marker_alpha.setVisible(is_scatter)
        self.plotter_inputs_widget.marker_alpha_spinbox.setVisible(is_scatter)

        # Contour plot elements
        self.plotter_inputs_widget.label_contour_levels.setVisible(is_contour)
        self.plotter_inputs_widget.contour_levels_spinbox.setVisible(
            is_contour
        )
        self.plotter_inputs_widget.label_contour_linewidth.setVisible(
            is_contour
        )
        self.plotter_inputs_widget.contour_linewidth_spinbox.setVisible(
            is_contour
        )
        self.plotter_inputs_widget.label_contour_layer_settings.setVisible(
            is_contour
        )
        self.plotter_inputs_widget.contour_layer_settings_button.setVisible(
            is_contour
        )

        self._update_contour_controls_visibility()
        self._refresh_main_colormap_control_for_mode()

        if not self._updating_settings:
            old_plot_type = getattr(self, '_current_plot_type', None)
            if new_plot_type != old_plot_type:
                self._current_plot_type = new_plot_type
                self._connect_active_artist_signals()
                self.switch_plot_type(new_plot_type)

    def _on_marker_size_changed(self, value):
        self._update_setting_in_metadata('marker_size', value)
        if not self._updating_settings and self.plot_type == 'SCATTER':
            self.canvas_widget.artists['SCATTER'].size = value
            self.canvas_widget.figure.canvas.draw_idle()

    def _on_marker_alpha_changed(self, value):
        self._update_setting_in_metadata('marker_alpha', value)
        if not self._updating_settings and self.plot_type == 'SCATTER':
            self.canvas_widget.artists['SCATTER'].alpha = value
            self.canvas_widget.figure.canvas.draw_idle()

    def _on_contour_levels_changed(self, value):
        self._update_setting_in_metadata('contour_levels', value)
        if not self._updating_settings and self.plot_type == 'CONTOUR':
            self.plot()

    def _on_contour_linewidth_changed(self, value):
        self._update_setting_in_metadata('contour_linewidth', value)
        if not self._updating_settings and self.plot_type == 'CONTOUR':
            self.plot()

    def _on_contour_layer_settings_clicked(self):
        selected_names = self.get_selected_layer_names()
        if len(selected_names) <= 1:
            notifications.show_info(
                "Contour layer settings are used when multiple layers are selected."
            )
            return

        current_layer_colors = {
            k: tuple(v)
            for k, v in (self._contour_layer_colors or {}).items()
            if k in selected_names
        }
        current_assignments = {
            k: int(v)
            for k, v in (self._contour_group_assignments or {}).items()
            if k in selected_names
        }
        current_group_colors = {
            int(k): tuple(v)
            for k, v in (self._contour_group_colors or {}).items()
        }
        current_group_names = {
            int(k): str(v)
            for k, v in (self._contour_group_names or {}).items()
        }
        current_layer_styles = {
            k: dict(v)
            for k, v in (self._contour_layer_styles or {}).items()
            if k in selected_names
        }
        current_group_styles = {
            int(k): dict(v)
            for k, v in (self._contour_group_styles or {}).items()
        }

        merged_colormap = self._contour_multi_layer_colormap
        if not merged_colormap:
            merged_colormap = (
                self.plotter_inputs_widget.colormap_combobox.currentText()
            )

        dialog = ContourLayerSettingsDialog(
            display_mode=self._contour_display_mode,
            merged_colormap=merged_colormap,
            merged_style=self._contour_merged_style,
            merged_color=self._contour_merged_color,
            show_legend=self._contour_show_legend,
            layer_labels=selected_names,
            group_assignments=current_assignments,
            layer_colors=current_layer_colors,
            group_colors=current_group_colors,
            group_names=current_group_names,
            layer_styles=current_layer_styles,
            group_styles=current_group_styles,
            available_colormaps=list(colormaps.ALL_COLORMAPS.keys()),
            parent=self,
        )

        if dialog.exec_() != QDialog.Accepted:
            return

        self._contour_display_mode = dialog.get_display_mode()
        self._contour_multi_layer_colormap = dialog.get_merged_colormap()
        self._contour_merged_style = dialog.get_merged_style()
        self._contour_merged_color = dialog.get_merged_color()
        self._contour_show_legend = dialog.get_show_legend()
        self._contour_layer_styles = dialog.get_layer_styles()
        self._contour_group_styles = dialog.get_group_styles()
        self._contour_layer_colors = dialog.get_layer_colors()
        self._contour_group_assignments = dialog.get_group_assignments()
        self._contour_group_colors = dialog.get_group_colors()
        self._contour_group_names = dialog.get_group_names()

        self._update_setting_in_metadata(
            'contour_display_mode', self._contour_display_mode
        )
        self._update_setting_in_metadata(
            'contour_multi_layer_colormap', self._contour_multi_layer_colormap
        )
        self._update_setting_in_metadata(
            'contour_merged_style', self._contour_merged_style
        )
        self._update_setting_in_metadata(
            'contour_merged_color', self._contour_merged_color
        )
        self._update_setting_in_metadata(
            'contour_layer_colors', self._contour_layer_colors
        )
        self._update_setting_in_metadata(
            'contour_group_assignments', self._contour_group_assignments
        )
        self._update_setting_in_metadata(
            'contour_group_colors', self._contour_group_colors
        )
        self._update_setting_in_metadata(
            'contour_group_names', self._contour_group_names
        )
        self._update_setting_in_metadata(
            'contour_layer_styles', self._contour_layer_styles
        )
        self._update_setting_in_metadata(
            'contour_group_styles', self._contour_group_styles
        )
        self._update_setting_in_metadata(
            'contour_show_legend', self._contour_show_legend
        )

        if self.plot_type == 'CONTOUR':
            self.plot()

    def _update_single_contour_color_button(self):
        if self.plot_type == 'HISTOGRAM2D':
            rgb = self._normalize_rgb(self._histogram_color)
        else:
            rgb = self._normalize_rgb(self._single_contour_color)
        self.plotter_inputs_widget.contour_single_color_button.setStyleSheet(
            "background-color: rgb("
            f"{int(rgb[0] * 255)}, {int(rgb[1] * 255)}, {int(rgb[2] * 255)}"
            ");"
        )

    def _populate_main_colormap_combobox(
        self, include_select_color=True, selected=None
    ):
        populate_colormap_combobox(
            self.plotter_inputs_widget.colormap_combobox,
            include_select_color=include_select_color,
            selected=selected,
        )

    def _refresh_main_colormap_control_for_mode(self):
        is_histogram = self.plot_type == 'HISTOGRAM2D'
        is_contour = self.plot_type == 'CONTOUR'
        single_layer = not self._has_multiple_selected_layers()
        is_contour_single = is_contour and single_layer

        include_select_color = is_histogram or is_contour_single
        if is_histogram and self._histogram_style == 'solid':
            selected = "Select color..."
        elif is_histogram:
            selected = self._histogram_colormap_name
        elif is_contour_single and self._single_contour_style == 'solid':
            selected = "Select color..."
        elif is_contour_single:
            selected = self._single_contour_colormap
        else:
            selected = self.histogram_colormap

        self._populate_main_colormap_combobox(
            include_select_color=include_select_color,
            selected=selected,
        )

        self.plotter_inputs_widget.contour_single_color_button.setVisible(
            (is_histogram and self._histogram_style == 'solid')
            or (is_contour_single and self._single_contour_style == 'solid')
        )
        self._update_single_contour_color_button()

    def _populate_main_colormap_combobox(
        self, include_select_color=True, selected=None
    ):
        populate_colormap_combobox(
            self.plotter_inputs_widget.colormap_combobox,
            include_select_color=include_select_color,
            selected=selected,
        )

    def _on_single_contour_color_clicked(self):
        from qtpy.QtWidgets import QColorDialog

        color = QColorDialog.getColor(parent=self)
        if not color.isValid():
            return

        if self.plot_type == 'HISTOGRAM2D':
            self._histogram_style = 'solid'
            self._histogram_color = color.getRgbF()[:3]
            self._update_setting_in_metadata(
                'histogram_style', self._histogram_style
            )
            self._update_setting_in_metadata(
                'histogram_color', self._histogram_color
            )
        else:
            self._single_contour_style = 'solid'
            self._single_contour_color = color.getRgbF()[:3]
            self._update_setting_in_metadata(
                'contour_single_style', self._single_contour_style
            )
            self._update_setting_in_metadata(
                'contour_single_color', self._single_contour_color
            )

        self._update_single_contour_color_button()

        if self.plot_type in ('CONTOUR', 'HISTOGRAM2D'):
            self.plot()

    def _has_multiple_selected_layers(self):
        return len(self.get_selected_layer_names()) > 1

    def _update_contour_controls_visibility(self):
        if self._is_closing or not self._has_plot_type_controls():
            return
        is_scatter = (
            self.plotter_inputs_widget.plot_type_combobox.currentText()
            == 'SCATTER'
        )
        is_contour = (
            self.plotter_inputs_widget.plot_type_combobox.currentText()
            == 'CONTOUR'
        )
        has_multiple = self._has_multiple_selected_layers()

        show_colormap = (not is_scatter) and not (is_contour and has_multiple)
        self.plotter_inputs_widget.label_3.setVisible(show_colormap)
        self._colormap_row_widget.setVisible(show_colormap)

        show_multi_layer_contour_controls = is_contour and has_multiple
        self._reflow_plot_settings_rows(show_multi_layer_contour_controls)
        self.plotter_inputs_widget.label_contour_layer_settings.setVisible(
            show_multi_layer_contour_controls
        )
        self.plotter_inputs_widget.contour_layer_settings_button.setVisible(
            show_multi_layer_contour_controls
        )
        self._refresh_main_colormap_control_for_mode()

    def _reflow_plot_settings_rows(self, show_multi_layer_contour_controls):
        """Reposition rows to avoid empty spacing when contour row is hidden."""
        layout = getattr(self, '_plotter_settings_layout', None)
        if layout is None:
            return

        if show_multi_layer_contour_controls:
            layout.addWidget(
                self.plotter_inputs_widget.label_contour_layer_settings, 3, 0
            )
            layout.addWidget(
                self.plotter_inputs_widget.contour_layer_settings_button, 3, 1
            )
        else:
            # Keep out of visible rows so histogram/scatter rows stay compact.
            layout.addWidget(
                self.plotter_inputs_widget.label_contour_layer_settings, 99, 0
            )
            layout.addWidget(
                self.plotter_inputs_widget.contour_layer_settings_button, 99, 1
            )

        row_shift = 1 if show_multi_layer_contour_controls else 0
        row_pairs = [
            (self.plotter_inputs_widget.label_3, self._colormap_row_widget, 3),
            (
                self.plotter_inputs_widget.label_4,
                self.plotter_inputs_widget.number_of_bins_spinbox,
                4,
            ),
            (
                self.plotter_inputs_widget.label_7,
                self.plotter_inputs_widget.log_scale_checkbox,
                5,
            ),
            (
                self.plotter_inputs_widget.label_marker_size,
                self.plotter_inputs_widget.marker_size_spinbox,
                6,
            ),
            (
                self.plotter_inputs_widget.label_marker_color,
                self.plotter_inputs_widget.marker_color_button,
                7,
            ),
            (
                self.plotter_inputs_widget.label_marker_alpha,
                self.plotter_inputs_widget.marker_alpha_spinbox,
                8,
            ),
            (
                self.plotter_inputs_widget.label_contour_levels,
                self.plotter_inputs_widget.contour_levels_spinbox,
                9,
            ),
            (
                self.plotter_inputs_widget.label_contour_linewidth,
                self.plotter_inputs_widget.contour_linewidth_spinbox,
                10,
            ),
        ]
        for left_widget, right_widget, base_row in row_pairs:
            target_row = base_row + row_shift
            layout.addWidget(left_widget, target_row, 0)
            layout.addWidget(right_widget, target_row, 1)

    def _on_marker_color_clicked(self):
        from qtpy.QtWidgets import QColorDialog

        color = QColorDialog.getColor(parent=self)
        if color.isValid():
            hex_color = color.name()
            self.plotter_inputs_widget.marker_color_button.setStyleSheet(
                f"background-color: {hex_color};"
            )
            self._marker_color = hex_color
            self._update_setting_in_metadata('marker_color', hex_color)
            if not self._updating_settings:
                self._update_scatter_colormap()

    def _update_scatter_colormap(self):
        from matplotlib.colors import ListedColormap

        current_color = getattr(self, '_marker_color', '#1f77b4')

        new_cmap = ListedColormap([current_color])

        self.canvas_widget.artists['SCATTER'].overlay_colormap = new_cmap
        self.canvas_widget.artists['SCATTER']._colorize(
            self.canvas_widget.artists['SCATTER'].color_indices
        )
        self.canvas_widget.figure.canvas.draw_idle()

    def _on_colormap_changed(self):
        """Callback for colormap change."""
        colormap = self.plotter_inputs_widget.colormap_combobox.currentText()
        if self.plot_type == 'HISTOGRAM2D':
            if colormap == "Select color...":
                self._histogram_style = 'solid'
                self._update_setting_in_metadata(
                    'histogram_style', self._histogram_style
                )
                self.plotter_inputs_widget.contour_single_color_button.setVisible(
                    True
                )
                self._update_single_contour_color_button()
            else:
                self._histogram_style = 'colormap'
                self._histogram_colormap_name = colormap
                self._update_setting_in_metadata(
                    'histogram_style', self._histogram_style
                )
                self._update_setting_in_metadata(
                    'colormap', self._histogram_colormap_name
                )
                self.plotter_inputs_widget.contour_single_color_button.setVisible(
                    False
                )

            if not self._updating_settings:
                self.refresh_current_plot()
            return

        if (
            self.plot_type == 'CONTOUR'
            and not self._has_multiple_selected_layers()
        ):
            if colormap == "Select color...":
                self._single_contour_style = 'solid'
                self._update_setting_in_metadata(
                    'contour_single_style', self._single_contour_style
                )
                self.plotter_inputs_widget.contour_single_color_button.setVisible(
                    True
                )
                self._update_single_contour_color_button()
            else:
                self._single_contour_style = 'colormap'
                self._single_contour_colormap = colormap
                self._update_setting_in_metadata(
                    'contour_single_style', self._single_contour_style
                )
                self._update_setting_in_metadata(
                    'contour_single_colormap', self._single_contour_colormap
                )
                self.plotter_inputs_widget.contour_single_color_button.setVisible(
                    False
                )

            if not self._updating_settings:
                self.refresh_current_plot()
            return

        self._update_setting_in_metadata('colormap', colormap)
        if not self._updating_settings and (
            self.plot_type == 'HISTOGRAM2D' or self.plot_type == 'CONTOUR'
        ):
            self.refresh_current_plot()

    def _on_bins_changed(self, value):
        """Callback for bins change.

        Debounced to avoid excessive updates while user is still adjusting.
        The actual processing happens in _process_bins_change.
        """
        self._update_setting_in_metadata('number_of_bins', value)
        if not self._updating_settings:
            # Start/restart timer - will fire after user stops changing value
            self._bins_timer.stop()
            self._bins_timer.start()

    def _process_bins_change(self):
        """Process bins change after debounce timer expires."""
        if self._is_closing or not self._has_plot_type_controls():
            return
        self.refresh_current_plot()

    def _refresh_plot_safely_for_log_scale(self):
        """Refresh plot while suppressing known log-normalization warning.

        The warning can also be emitted during transitions when toggling
        log scale off (artist state updates happen within a refresh cycle),
        so we filter it unconditionally for this refresh path.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Log normalization applied to color indices*",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message="Attempt to set non-positive ylim on a log-scaled axis will be ignored*",
                category=UserWarning,
            )
            self.refresh_current_plot()

    def _on_log_scale_changed(self, state):
        """Callback for log scale change."""
        self._update_setting_in_metadata('log_scale', bool(state))
        if not self._updating_settings and self.plot_type == 'HISTOGRAM2D':
            self._refresh_plot_safely_for_log_scale()

    def _on_white_background_changed(self, state):
        """Callback for white background checkbox change."""
        self._update_setting_in_metadata('white_background', bool(state))
        if not self._updating_settings:
            self.set_axes_labels()
            self._update_plot_bg_color()

            if self.toggle_semi_circle:
                self._update_semi_circle_plot(self.canvas_widget.axes)
            else:
                self._update_polar_plot(self.canvas_widget.axes, visible=True)

            self.canvas_widget.figure.canvas.draw_idle()
            self._refresh_plot_safely_for_log_scale()

    def on_white_background_changed(self):
        """Callback function when the white background checkbox is toggled."""
        self.set_axes_labels()
        if self.toggle_semi_circle:
            self._update_semi_circle_plot(self.canvas_widget.axes)
        else:
            self._update_polar_plot(self.canvas_widget.axes, visible=True)
        self.canvas_widget.figure.canvas.draw_idle()

        self._refresh_plot_safely_for_log_scale()

    @property
    def white_background(self):
        """Gets the white background value from the white background checkbox.

        Returns
        -------
        bool
            The white background value.
        """
        return self.plotter_inputs_widget.white_background_checkbox.isChecked()

    @white_background.setter
    def white_background(self, value: bool):
        """Sets the white background value from the white background checkbox."""
        self.plotter_inputs_widget.white_background_checkbox.setChecked(value)
        self.set_axes_labels()
        self._refresh_plot_safely_for_log_scale()

    def _create_calibration_tab(self):
        """Create the Calibration tab."""
        self.calibration_tab = CalibrationWidget(self.viewer, parent=self)
        self.tab_widget.addTab(self.calibration_tab, "Calibration")
        self.calibration_tab.calibration_widget.frequency_input.editingFinished.connect(
            lambda: self._broadcast_frequency_value_across_tabs(
                self.calibration_tab.calibration_widget.frequency_input.text()
            )
        )

    def _create_filter_tab(self):
        """Create the Filtering and Thresholding tab."""
        self.filter_tab = FilterWidget(self.viewer, parent=self)
        self.tab_widget.addTab(self.filter_tab, "Filter")

    def _create_selection_tab(self):
        """Create the Cursor selection tab."""
        self.selection_tab = SelectionWidget(self.viewer, parent=self)
        self.tab_widget.addTab(self.selection_tab, "Selection")

    def _create_components_tab(self):
        """Create the Components tab."""
        self.components_tab = ComponentsWidget(self.viewer, parent=self)
        self.tab_widget.addTab(self.components_tab, "Components")

        # Wrap the histogram in a HistogramDockWidget and add to shared stack
        self.components_histogram_dock_widget = HistogramDockWidget(
            self.components_tab.histogram_widget,
            title="Components Histogram & Statistics",
        )

        # Insert component selector combobox at the top of the dock widget
        dock_layout = self.components_histogram_dock_widget.layout()
        component_selector = QWidget()
        selector_layout = QHBoxLayout(component_selector)
        selector_layout.setContentsMargins(4, 4, 4, 0)
        selector_layout.addWidget(QLabel("Component:"))
        selector_layout.addWidget(
            self.components_tab.histogram_component_combobox, 1
        )
        dock_layout.insertWidget(0, component_selector)

        self._components_hist_page_idx = self._histogram_stack.addWidget(
            self.components_histogram_dock_widget
        )

        # Create the matching statistics dock and link it
        self.components_statistics_dock_widget = StatisticsDockWidget(
            self.components_tab.histogram_widget,
            title="Components Statistics",
        )
        self._components_stats_page_idx = self._statistics_stack.addWidget(
            self.components_statistics_dock_widget
        )
        self.components_histogram_dock_widget.link_statistics_dock(
            self.components_statistics_dock_widget
        )

    def _create_phasor_mapping_tab(self):
        """Create the Phasor Mapping tab."""
        self.phasor_mapping_tab = PhasorMappingWidget(self.viewer, parent=self)
        self.tab_widget.addTab(self.phasor_mapping_tab, "Phasor Mapping")

        # Wrap the histogram in a HistogramDockWidget and add to shared stack
        self.phasor_map_histogram_dock_widget = HistogramDockWidget(
            self.phasor_mapping_tab.histogram_widget,
            title="Output Histogram & Statistics",
        )

        self._phasor_map_hist_page_idx = self._histogram_stack.addWidget(
            self.phasor_map_histogram_dock_widget
        )

        # Create the matching statistics dock and link it
        self.phasor_map_statistics_dock_widget = StatisticsDockWidget(
            self.phasor_mapping_tab.histogram_widget,
            title="Output Statistics",
        )
        self._phasor_map_stats_page_idx = self._statistics_stack.addWidget(
            self.phasor_map_statistics_dock_widget
        )
        self.phasor_map_histogram_dock_widget.link_statistics_dock(
            self.phasor_map_statistics_dock_widget
        )
        self.phasor_mapping_tab.frequency_input.editingFinished.connect(
            lambda: self._broadcast_frequency_value_across_tabs(
                self.phasor_mapping_tab.frequency_input.text()
            )
        )
        self.phasor_mapping_tab.outputTypeChanged.connect(
            lambda _: self._update_histogram_dock_visibility(
                self.tab_widget.currentWidget()
            )
        )

    def _create_fret_tab(self):
        """Create the FRET tab."""
        self.fret_tab = FretWidget(self.viewer, parent=self)
        self.tab_widget.addTab(self.fret_tab, "FRET")

        # Wrap the histogram in a HistogramDockWidget and add to shared stack
        self.fret_histogram_dock_widget = HistogramDockWidget(
            self.fret_tab.histogram_widget,
            title="FRET Histogram & Statistics",
        )
        self._fret_hist_page_idx = self._histogram_stack.addWidget(
            self.fret_histogram_dock_widget
        )

        # Create the matching statistics dock and link it
        self.fret_statistics_dock_widget = StatisticsDockWidget(
            self.fret_tab.histogram_widget,
            title="FRET Statistics",
        )
        self._fret_stats_page_idx = self._statistics_stack.addWidget(
            self.fret_statistics_dock_widget
        )
        self.fret_histogram_dock_widget.link_statistics_dock(
            self.fret_statistics_dock_widget
        )

        self.fret_tab.frequency_input.editingFinished.connect(
            lambda: self._broadcast_frequency_value_across_tabs(
                self.fret_tab.frequency_input.text()
            )
        )
        self.harmonic_spinbox.valueChanged.connect(
            self.fret_tab._on_harmonic_changed
        )

    @property
    def harmonic(self):
        """Gets or sets the harmonic value from the harmonic spinbox.

        Returns
        -------
        int
            The harmonic value.
        """
        return self.harmonic_spinbox.value()

    @harmonic.setter
    def harmonic(self, value: int):
        """Sets the harmonic value from the harmonic spinbox."""
        if value < 1:
            notifications.WarningNotification(
                "Harmonic value should be greater than 0. Setting to 1."
            )
            value = 1
        self.harmonic_spinbox.setValue(value)

    @property
    def toggle_semi_circle(self):
        """Gets the display semi circle value from the semi circle checkbox.

        Returns
        -------
        bool
            The display semi circle value.
        """
        return self.plotter_inputs_widget.semi_circle_checkbox.isChecked()

    @toggle_semi_circle.setter
    def toggle_semi_circle(self, value: bool):
        """Sets the display semi circle value from the semi circle checkbox."""
        self.plotter_inputs_widget.semi_circle_checkbox.setChecked(value)
        if value:
            self._update_polar_plot(self.canvas_widget.axes, visible=False)
            self._update_semi_circle_plot(self.canvas_widget.axes)
        else:
            self._update_semi_circle_plot(
                self.canvas_widget.axes, visible=False
            )
            self._update_polar_plot(self.canvas_widget.axes)
        self._enforce_axes_aspect()

    def on_toggle_semi_circle(self, state):
        """Callback function when the semi circle checkbox is toggled.

        This function updates the `toggle_semi_circle` attribute with the
        checked status of the checkbox. And it displays either the universal
        semi-circle or the full polar plot in the canvas widget.

        """
        self.toggle_semi_circle = state

    def _update_polar_plot(self, ax, visible=True, alpha=0.5):
        """Generate the polar plot in the canvas widget."""
        line_color = 'black' if self.white_background else 'white'

        if len(self.polar_plot_artist_list) > 0:
            for artist in self.polar_plot_artist_list:
                artist.set_visible(visible)
                artist.set_alpha(alpha)
                artist.set_color(line_color)
        else:
            self.polar_plot_artist_list.append(
                ax.add_line(
                    Line2D(
                        [-1, 1],
                        [0, 0],
                        linestyle='-',
                        linewidth=1,
                        color=line_color,
                    )
                )
            )
            self.polar_plot_artist_list.append(
                ax.add_line(
                    Line2D(
                        [0, 0],
                        [-1, 1],
                        linestyle='-',
                        linewidth=1,
                        color=line_color,
                    )
                )
            )
            circle = Circle((0, 0), 1, fill=False, color=line_color)
            self.polar_plot_artist_list.append(ax.add_patch(circle))
            for r in (1 / 3, 2 / 3):
                circle = Circle((0, 0), r, fill=False, color=line_color)
                self.polar_plot_artist_list.append(ax.add_patch(circle))
            for a in (3, 6):
                x = math.cos(math.pi / a)
                y = math.sin(math.pi / a)
                self.polar_plot_artist_list.append(
                    ax.add_line(
                        Line2D(
                            [-x, x],
                            [-y, y],
                            linestyle=':',
                            linewidth=0.5,
                            color=line_color,
                        )
                    )
                )
                self.polar_plot_artist_list.append(
                    ax.add_line(
                        Line2D(
                            [-x, x],
                            [y, -y],
                            linestyle=':',
                            linewidth=0.5,
                            color=line_color,
                        )
                    )
                )
        return ax

    def _update_semi_circle_plot(self, ax, visible=True, alpha=0.5, zorder=3):
        """Generate FLIM universal semi-circle plot."""
        line_color = 'black' if self.white_background else 'white'

        if len(self.semi_circle_plot_artist_list) > 0:
            for artist in self.semi_circle_plot_artist_list:
                artist.remove()
            self.semi_circle_plot_artist_list.clear()

        if visible:
            angles = np.linspace(0, np.pi, 180)
            x = (np.cos(angles) + 1) / 2
            y = np.sin(angles) / 2
            self.semi_circle_plot_artist_list.append(
                ax.plot(
                    x,
                    y,
                    color=line_color,
                    alpha=alpha,
                    visible=visible,
                    zorder=zorder,
                )[0]
            )

            self._add_lifetime_ticks_to_semicircle(ax, visible, alpha, zorder)

        return ax

    def _add_lifetime_ticks_to_semicircle(
        self, ax, visible=True, alpha=0.5, zorder=3
    ):
        """Add lifetime ticks to the semicircle plot based on frequency."""
        frequency = self._get_frequency_from_layer()
        if frequency is None:
            return

        effective_frequency = frequency * self.harmonic

        tick_color = 'black' if self.white_background else 'darkgray'

        lifetimes = [0.0]

        # Add powers of 2 that result in S coordinates >= 0.18
        for t in range(-8, 32):
            lifetime_val = 2**t
            try:
                g_pos, s_pos = phasor_from_lifetime(
                    effective_frequency, lifetime_val
                )
                if s_pos >= 0.18:
                    lifetimes.append(lifetime_val)
            except Exception:  # noqa: BLE001
                continue

        for _i, lifetime in enumerate(lifetimes):
            if lifetime == 0:
                g_pos, s_pos = 1.0, 0.0
            else:
                g_pos, s_pos = phasor_from_lifetime(
                    effective_frequency, lifetime
                )

            center_x, center_y = 0.5, 0.0
            dx = g_pos - center_x
            dy = s_pos - center_y
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                dx_norm = dx / length
                dy_norm = dy / length
            else:
                dx_norm = 1.0
                dy_norm = 0.0

            tick_length = 0.03
            tick_start_x = g_pos
            tick_start_y = s_pos
            tick_end_x = g_pos + tick_length * dx_norm
            tick_end_y = s_pos + tick_length * dy_norm

            tick_line = ax.plot(
                [tick_start_x, tick_end_x],
                [tick_start_y, tick_end_y],
                color=tick_color,
                linewidth=1.5,
                alpha=alpha,
                visible=visible,
                zorder=zorder + 1,
                clip_on=True,
            )[0]
            tick_line.set_clip_path(ax.patch)
            self.semi_circle_plot_artist_list.append(tick_line)

            label_text = "0" if lifetime == 0 else f"{lifetime:g}"

            label_offset = 0.08
            label_x = g_pos + label_offset * dx_norm
            label_y = s_pos + label_offset * dy_norm

            text_color = tick_color

            label = ax.text(
                label_x,
                label_y,
                label_text,
                fontsize=8,
                ha='center',
                va='center',
                color=text_color,
                alpha=alpha,
                visible=visible,
                zorder=zorder + 1,
                clip_on=True,
            )
            label.set_clip_path(ax.patch)
            self.semi_circle_plot_artist_list.append(label)

    def _get_frequency_from_layer(self):
        """Get frequency from the current layer's metadata."""
        layer_name = (
            self.image_layer_with_phasor_features_combobox.currentText()
        )
        if layer_name == "":
            return None
        layer = self.viewer.layers[layer_name]
        if "settings" in layer.metadata:
            settings = layer.metadata["settings"]
            if "frequency" in settings:
                try:
                    return float(settings["frequency"])
                except (ValueError, TypeError):
                    return None

        return None

    def _sync_frequency_inputs_from_metadata(self):
        """Sync the frequency widget input fields with metadata."""
        layer_name = (
            self.image_layer_with_phasor_features_combobox.currentText()
        )
        if not layer_name:
            self._broadcast_frequency_value_across_tabs("")
            return

        layer = self.viewer.layers[layer_name]
        settings = layer.metadata.get('settings', {})
        frequency = settings.get('frequency', None)

        if frequency is not None:
            self._broadcast_frequency_value_across_tabs(str(frequency))
        else:
            self._broadcast_frequency_value_across_tabs("")

    def _broadcast_frequency_value_across_tabs(self, value):
        """
        Broadcast the frequency value to all relevant input fields and update semicircle.
        """
        self.calibration_tab.calibration_widget.frequency_input.blockSignals(
            True
        )
        self.phasor_mapping_tab.frequency_input.blockSignals(True)
        self.fret_tab.frequency_input.blockSignals(True)

        try:
            if value and value.strip():
                freq_val = float(value)
                layer_name = (
                    self.image_layer_with_phasor_features_combobox.currentText()
                )
                if layer_name:
                    layer = self.viewer.layers[layer_name]
                    update_frequency_in_metadata(layer, freq_val)
        except (ValueError, TypeError):
            pass

        self.calibration_tab.calibration_widget.frequency_input.setText(value)
        self.phasor_mapping_tab.frequency_input.setText(value)
        self.fret_tab.frequency_input.setText(value)

        self.calibration_tab.calibration_widget.frequency_input.blockSignals(
            False
        )
        self.phasor_mapping_tab.frequency_input.blockSignals(False)
        self.fret_tab.frequency_input.blockSignals(False)

        self.components_tab._update_lifetime_inputs_visibility()

        if self.toggle_semi_circle:
            self._update_semi_circle_plot(self.canvas_widget.axes)
            self.canvas_widget.figure.canvas.draw_idle()

    def _redefine_axes_limits(self, ensure_full_circle_displayed=True):
        """
        Redefine axes limits based on the data plotted in the canvas widget.

        Parameters
        ----------
        ensure_full_circle_displayed : bool, optional
            Whether to ensure the full circle is displayed in the canvas widget.
            By default True.
        """
        if self._user_axes_limits is not None:
            xlim, ylim = self._user_axes_limits
            self.canvas_widget.axes.set_xlim(xlim)
            self.canvas_widget.axes.set_ylim(ylim)
            self.canvas_widget.figure.canvas.draw_idle()
            return
        if self.toggle_semi_circle:
            self.canvas_widget.axes.set_xlim([-0.1, 1.1])
            self.canvas_widget.axes.set_ylim([-0.1, 0.6])

            if (
                hasattr(self.canvas_widget, 'toolbar')
                and self.canvas_widget.toolbar
            ):
                self.canvas_widget.toolbar.update()
                self.canvas_widget.toolbar.push_current()

            self.canvas_widget.figure.canvas.draw_idle()
            return

        circle_plot_limits = [-1, 1, -1, 1]  # xmin, xmax, ymin, ymax
        if self.canvas_widget.artists['HISTOGRAM2D'].histogram is not None:
            x_edges = self.canvas_widget.artists['HISTOGRAM2D'].histogram[1]
            y_edges = self.canvas_widget.artists['HISTOGRAM2D'].histogram[2]
            plotted_data_limits = [
                x_edges[0],
                x_edges[-1],
                y_edges[0],
                y_edges[-1],
            ]
        else:
            plotted_data_limits = circle_plot_limits
        if not ensure_full_circle_displayed:
            circle_plot_limits = plotted_data_limits

        x_range = np.amax(
            [plotted_data_limits[1], circle_plot_limits[1]]
        ) - np.amin([plotted_data_limits[0], circle_plot_limits[0]])
        y_range = np.amax(
            [plotted_data_limits[3], circle_plot_limits[3]]
        ) - np.amin([plotted_data_limits[2], circle_plot_limits[2]])
        xlim_0 = (
            np.amin([plotted_data_limits[0], circle_plot_limits[0]])
            - 0.1 * x_range
        )
        xlim_1 = (
            np.amax([plotted_data_limits[1], circle_plot_limits[1]])
            + 0.1 * x_range
        )
        ylim_0 = (
            np.amin([plotted_data_limits[2], circle_plot_limits[2]])
            - 0.1 * y_range
        )
        ylim_1 = (
            np.amax([plotted_data_limits[3], circle_plot_limits[3]])
            + 0.1 * y_range
        )

        self.canvas_widget.axes.set_ylim([ylim_0, ylim_1])
        self.canvas_widget.axes.set_xlim([xlim_0, xlim_1])

        if (
            hasattr(self.canvas_widget, 'toolbar')
            and self.canvas_widget.toolbar
        ):
            self.canvas_widget.toolbar.update()
            self.canvas_widget.toolbar.push_current()

        self.canvas_widget.figure.canvas.draw_idle()

    def _update_plot_bg_color(self, color=None):
        """Change the background color of the canvas widget.

        Parameters
        ----------
        color : str, optional
            The color to set the background, by default None.
            If None, the background will be set based on the white_background
            checkbox state.
        """
        if color is None:
            color = "white" if self.white_background else "none"

        if color == "none":
            self.canvas_widget.axes.set_facecolor('none')
            self.canvas_widget.figure.patch.set_facecolor('none')
        else:
            self.canvas_widget.axes.set_facecolor(color)
            self.canvas_widget.figure.patch.set_facecolor('none')

        self.canvas_widget.figure.canvas.draw_idle()

    @property
    def plot_type(self):
        """Gets or sets the plot type from the plot type combobox.

        Returns
        -------
        str
            The plot type.
        """
        return self._get_plot_type_safe()

    @plot_type.setter
    def plot_type(self, value):
        """Sets the plot type from the plot type combobox."""
        if self._is_closing or not self._has_plot_type_controls():
            return
        self.plotter_inputs_widget.plot_type_combobox.setCurrentText(value)

    def _has_plot_type_controls(self):
        """Return True if the plot-type controls are valid Qt objects."""
        try:
            combo = self.plotter_inputs_widget.plot_type_combobox
            combo.currentText()
            return True
        except (AttributeError, RuntimeError):
            return False

    def _get_plot_type_safe(self, default='HISTOGRAM2D'):
        """Safely read plot type while controls may be tearing down."""
        try:
            return self.plotter_inputs_widget.plot_type_combobox.currentText()
        except (AttributeError, RuntimeError):
            return getattr(self, '_current_plot_type', default)

    @property
    def histogram_colormap(self):
        """Gets or sets the histogram colormap from the colormap combobox.

        Returns
        -------
        str
            The colormap name.
        """
        value = self.plotter_inputs_widget.colormap_combobox.currentText()
        if value == "Select color...":
            if self.plot_type == 'HISTOGRAM2D':
                return self._histogram_colormap_name
            return self._single_contour_colormap
        return value

    @histogram_colormap.setter
    def histogram_colormap(self, colormap: str):
        """Sets the histogram colormap from the colormap combobox."""
        if colormap not in colormaps.ALL_COLORMAPS:
            notifications.WarningNotification(
                f"{colormap} is not a valid colormap. Setting to default colormap."
            )
            colormap = self._histogram_colormap.name
        self._histogram_colormap_name = colormap
        self.plotter_inputs_widget.colormap_combobox.setCurrentText(colormap)

    @property
    def histogram_bins(self):
        """Gets the histogram bins from the histogram bins spinbox.

        Returns
        -------
        int
            The histogram bins value.
        """
        return self.plotter_inputs_widget.number_of_bins_spinbox.value()

    @histogram_bins.setter
    def histogram_bins(self, value: int):
        """Sets the histogram bins from the histogram bins spinbox."""
        if value < 2:
            notifications.WarningNotification(
                "Number of bins should be greater than 1. Setting to 10."
            )
            value = 10
        self.plotter_inputs_widget.number_of_bins_spinbox.setValue(value)

    @property
    def histogram_log_scale(self):
        """Gets the histogram log scale from the histogram log scale checkbox.

        Returns
        -------
        bool
            The histogram log scale value.
        """
        return self.plotter_inputs_widget.log_scale_checkbox.isChecked()

    @histogram_log_scale.setter
    def histogram_log_scale(self, value: bool):
        """Sets the histogram log scale from the histogram log scale checkbox."""
        self.plotter_inputs_widget.log_scale_checkbox.setChecked(value)

    def _enforce_axes_aspect(self):
        """Ensure the axes aspect is set to 'box' after artist redraws."""
        self._redefine_axes_limits()
        self.canvas_widget.axes.set_aspect(1, adjustable='box')
        self.canvas_widget.figure.canvas.draw_idle()

    def _connect_selector_signals(self):
        """Connect selection applied signal from all selectors to enforce axes aspect."""
        for selector in self.canvas_widget.selectors.values():
            selector.selection_applied_signal.connect(
                self._enforce_axes_aspect
            )

    def _connect_active_artist_signals(self):
        """Connect signals for the currently active artist only."""
        self._disconnect_all_artist_signals()

        if self.plot_type == 'HISTOGRAM2D':
            self.canvas_widget.artists[
                'HISTOGRAM2D'
            ].color_indices_changed_signal.connect(
                self.selection_tab.manual_selection_changed
            )
        elif self.plot_type == 'SCATTER':
            self.canvas_widget.artists[
                'SCATTER'
            ].color_indices_changed_signal.connect(
                self.selection_tab.manual_selection_changed
            )

    def _disconnect_all_artist_signals(self):
        """Disconnect all artist signals to prevent conflicts."""
        with contextlib.suppress(TypeError, AttributeError):
            self.canvas_widget.artists[
                'SCATTER'
            ].color_indices_changed_signal.disconnect(
                self.selection_tab.manual_selection_changed
            )

        with contextlib.suppress(TypeError, AttributeError):
            self.canvas_widget.artists[
                'HISTOGRAM2D'
            ].color_indices_changed_signal.disconnect(
                self.selection_tab.manual_selection_changed
            )

    def reset_layer_choices(self):
        """Reset the image layer checkable combobox choices."""
        if getattr(self, '_resetting_layer_choices', False):
            return

        self._resetting_layer_choices = True

        try:
            # Store current selection
            previously_selected = self.get_selected_layer_names()
            mask_layer_combobox_current_text = (
                self.mask_layer_combobox.currentText()
            )

            self.image_layers_checkable_combobox.blockSignals(True)
            self.mask_layer_combobox.blockSignals(True)

            self.image_layers_checkable_combobox.clear()
            self.mask_layer_combobox.clear()

            layer_names = [
                layer.name
                for layer in self.viewer.layers
                if isinstance(layer, Image)
                and "G" in layer.metadata
                and "S" in layer.metadata
                and "G_original" in layer.metadata
                and "S_original" in layer.metadata
            ]
            mask_layer_names = [
                layer.name
                for layer in self.viewer.layers
                if isinstance(layer, (Labels, Shapes))
            ]

            # Add items to the checkable combobox
            for name in layer_names:
                # Check if this layer was previously selected
                checked = name in previously_selected
                self.image_layers_checkable_combobox.addItem(name, checked)

            # If no layers were previously selected and we have layers, select the first one
            if not previously_selected and layer_names:
                self.image_layers_checkable_combobox.setCheckedItems(
                    [layer_names[0]]
                )

            self.mask_layer_combobox.addItems(["None"] + mask_layer_names)

            # Check if previously selected mask layer was deleted
            mask_layer_was_deleted = (
                mask_layer_combobox_current_text != "None"
                and mask_layer_combobox_current_text not in mask_layer_names
            )

            if mask_layer_combobox_current_text in mask_layer_names:
                self.mask_layer_combobox.setCurrentText(
                    mask_layer_combobox_current_text
                )

            # Clean up mask assignments for deleted mask layers or removed image layers
            current_selected = (
                self.image_layers_checkable_combobox.checkedItems()
            )
            self._mask_assignments = {
                k: v
                for k, v in self._mask_assignments.items()
                if k in current_selected and v in mask_layer_names
            }

            self.image_layers_checkable_combobox.blockSignals(False)
            self.mask_layer_combobox.blockSignals(False)

            # Update display text after unblocking signals
            self.image_layers_checkable_combobox._update_display_text()

            # Update mask UI mode (combobox vs button)
            self._update_mask_ui_mode()
            self._update_contour_controls_visibility()

            # If mask layer was deleted, trigger the cleanup
            if mask_layer_was_deleted:
                # Re-apply current mask assignments (deleted ones already removed above)
                selected_layers = self.get_selected_layers()
                if selected_layers and len(current_selected) > 1:
                    # Multi-layer mode: re-apply per-layer assignments
                    all_assignments = {
                        name: self._mask_assignments.get(name, "None")
                        for name in current_selected
                    }
                    self._apply_mask_assignments(all_assignments)
                else:
                    self._on_mask_layer_changed("None")

            # Connect layer name change events (disconnect first to avoid duplicates)
            for layer_name in layer_names + mask_layer_names:
                layer = self.viewer.layers[layer_name]
                if (
                    isinstance(layer, Image)
                    and "phasor_features_labels_layer" in layer.metadata
                ):
                    with contextlib.suppress(TypeError, ValueError):
                        layer.events.name.disconnect(self.reset_layer_choices)
                    layer.events.name.connect(self.reset_layer_choices)
                if isinstance(layer, Shapes):
                    with contextlib.suppress(TypeError, ValueError):
                        layer.events.data.disconnect(
                            self._on_mask_data_changed
                        )
                    layer.events.data.connect(self._on_mask_data_changed)
                if isinstance(layer, Labels):
                    try:
                        layer.events.paint.disconnect(
                            self._on_mask_data_changed
                        )
                        layer.events.set_data.disconnect(
                            self._on_mask_data_changed
                        )
                    except (TypeError, ValueError):
                        pass  # Not connected, ignore
                    layer.events.paint.connect(self._on_mask_data_changed)
                    layer.events.set_data.connect(self._on_mask_data_changed)

            new_selected = self.get_selected_layer_names()
            if new_selected != previously_selected or (
                not previously_selected and new_selected
            ):
                self.on_image_layer_changed()
                self._sync_frequency_inputs_from_metadata()

        finally:
            self._resetting_layer_choices = False

    def on_image_layer_changed(self):
        """Handle changes to the image layer with phasor features.

        When multiple layers are selected, the primary (first) layer is used
        for metadata operations and analysis, while all selected layers
        contribute to the merged plot.

        This is the full handler used for initial load and layer list updates.
        For incremental changes, use _on_primary_layer_changed or _on_selection_changed.
        """
        if getattr(self, "_in_on_image_layer_changed", False):
            return
        self._in_on_image_layer_changed = True
        try:
            selected_layers = self.get_selected_layers()

            self._update_grid_view(selected_layers)

            layer_name = self.get_primary_layer_name()
            if layer_name == "":
                self._g_array = None
                self._s_array = None
                self._g_original_array = None
                self._s_original_array = None
                self._harmonics_array = None
                # Clear phasor data artists (histogram/scatter) but keep
                # the semicircle/polar plot and axes styling intact.
                for artist in self.canvas_widget.artists.values():
                    artist._remove_artists()
                self._remove_colorbar()
                # Clear tab-specific artists
                self._clear_all_tab_artists()
                # Redraw circle/semicircle and axes
                self.set_axes_labels()
                self._update_plot_bg_color()
                if self.toggle_semi_circle:
                    self._update_semi_circle_plot(self.canvas_widget.axes)
                else:
                    self._update_polar_plot(self.canvas_widget.axes)
                self._redefine_axes_limits()
                self.canvas_widget.figure.canvas.draw_idle()
                return

            layer = self.viewer.layers[layer_name]
            layer_metadata = layer.metadata

            # On a fresh layer load, reset any user-defined zoom so the
            # view auto-fits to the new data.
            self._user_axes_limits = None

            self._g_array = layer_metadata.get("G")
            self._s_array = layer_metadata.get("S")
            self._g_original_array = layer_metadata.get("G_original")
            self._s_original_array = layer_metadata.get("S_original")
            self._harmonics_array = layer_metadata.get("harmonics")

            if len(selected_layers) > 1:
                common_harmonics = self._get_common_harmonics(selected_layers)
                if common_harmonics is not None and len(common_harmonics) > 0:
                    min_harmonic = int(np.min(common_harmonics))
                    max_harmonic = int(np.max(common_harmonics))
                    self.harmonic_spinbox.setRange(min_harmonic, max_harmonic)
            elif self._harmonics_array is not None:
                self._harmonics_array = np.atleast_1d(self._harmonics_array)
                min_harmonic = int(np.min(self._harmonics_array))
                max_harmonic = int(np.max(self._harmonics_array))
                self.harmonic_spinbox.setRange(min_harmonic, max_harmonic)

            # Reset masks
            self._mask_assignments.clear()
            self.mask_layer_combobox.blockSignals(True)
            self.mask_layer_combobox.setCurrentText("None")
            self.mask_layer_combobox.blockSignals(False)
            self._update_mask_ui_mode()
            self._update_contour_controls_visibility()

            self._initialize_plot_settings_in_metadata(layer)

            self._restore_plot_settings_from_metadata()

            self._sync_frequency_inputs_from_metadata()

            if hasattr(self, 'filter_tab'):
                self.filter_tab._on_image_layer_changed()

            if hasattr(self, 'calibration_tab'):
                self.calibration_tab._on_image_layer_changed()

            if hasattr(self, 'selection_tab'):
                self.selection_tab._on_image_layer_changed()

            if hasattr(self, 'phasor_mapping_tab'):
                self.phasor_mapping_tab._on_image_layer_changed()

            if hasattr(self, 'components_tab'):
                self.components_tab._on_image_layer_changed()

            if hasattr(self, 'fret_tab'):
                self.fret_tab._on_image_layer_changed()

            self.plot()

            # Enforce tab-specific artist visibility based on current tab
            current_tab_index = self.tab_widget.currentIndex()
            self._on_tab_changed(current_tab_index)

        finally:
            self._in_on_image_layer_changed = False

    def _on_primary_layer_changed(self, new_primary_name):
        """Handle changes to the primary layer only.

        Updates tab UIs to reflect the new primary layer's settings,
        but does NOT automatically run analyses, only restores UI state.

        Parameters
        ----------
        new_primary_name : str
            The name of the new primary layer.
        """
        if getattr(self, "_in_on_primary_layer_changed", False):
            return
        self._in_on_primary_layer_changed = True
        try:
            selected_layers = self.get_selected_layers()
            self._update_grid_view(selected_layers)

            if not new_primary_name:
                self._g_array = None
                self._s_array = None
                self._g_original_array = None
                self._s_original_array = None
                self._harmonics_array = None
                for artist in self.canvas_widget.artists.values():
                    artist._remove_artists()
                self._remove_colorbar()
                self._clear_all_tab_artists()
                self.set_axes_labels()
                self._update_plot_bg_color()
                if self.toggle_semi_circle:
                    self._update_semi_circle_plot(self.canvas_widget.axes)
                else:
                    self._update_polar_plot(self.canvas_widget.axes)
                self._redefine_axes_limits()
                self.canvas_widget.figure.canvas.draw_idle()
                return

            layer = self.viewer.layers[new_primary_name]
            layer_metadata = layer.metadata

            self._g_array = layer_metadata.get("G")
            self._s_array = layer_metadata.get("S")
            self._g_original_array = layer_metadata.get("G_original")
            self._s_original_array = layer_metadata.get("S_original")
            self._harmonics_array = layer_metadata.get("harmonics")

            if len(selected_layers) > 1:
                common_harmonics = self._get_common_harmonics(selected_layers)
                if common_harmonics is not None and len(common_harmonics) > 0:
                    min_harmonic = int(np.min(common_harmonics))
                    max_harmonic = int(np.max(common_harmonics))
                    self.harmonic_spinbox.setRange(min_harmonic, max_harmonic)
            elif self._harmonics_array is not None:
                self._harmonics_array = np.atleast_1d(self._harmonics_array)
                min_harmonic = int(np.min(self._harmonics_array))
                max_harmonic = int(np.max(self._harmonics_array))
                self.harmonic_spinbox.setRange(min_harmonic, max_harmonic)

            # Reset masks
            self._mask_assignments.clear()
            self.mask_layer_combobox.blockSignals(True)
            self.mask_layer_combobox.setCurrentText("None")
            self.mask_layer_combobox.blockSignals(False)
            self._update_mask_ui_mode()
            self._update_contour_controls_visibility()

            self._initialize_plot_settings_in_metadata(layer)
            self._restore_plot_settings_from_metadata()

            if hasattr(self, 'filter_tab'):
                self.filter_tab._on_image_layer_changed()

            if hasattr(self, 'calibration_tab'):
                self.calibration_tab._on_image_layer_changed()

            if hasattr(self, 'selection_tab'):
                self.selection_tab._on_image_layer_changed()

            if hasattr(self, 'phasor_mapping_tab'):
                self.phasor_mapping_tab._on_image_layer_changed()

            if hasattr(self, 'components_tab'):
                self.components_tab._on_image_layer_changed()

            if hasattr(self, 'fret_tab'):
                self.fret_tab._on_image_layer_changed()

            self.plot()

            current_tab_index = self.tab_widget.currentIndex()
            self._on_tab_changed(current_tab_index)

        finally:
            self._in_on_primary_layer_changed = False

    def _on_selection_changed(self):
        """Handle changes to layer selection (adding/removing layers).

        Debounced to avoid excessive updates while user is still selecting.
        The actual processing happens in _process_layer_selection_change.
        """
        if self._is_closing or not self._has_plot_type_controls():
            return
        self._preserve_plot_type_on_restore = self.plot_type == 'CONTOUR'
        self._update_contour_controls_visibility()
        self._layer_selection_timer.stop()
        self._layer_selection_timer.start()

    def _process_layer_selection_change(self):
        """Process layer selection change after debounce timer expires.

        Only updates the phasor plot. Tab UIs are not updated unless
        the primary layer changed (which triggers _on_primary_layer_changed).
        """
        if self._is_closing or not self._has_plot_type_controls():
            return
        if getattr(self, "_in_on_selection_changed", False):
            return
        self._in_on_selection_changed = True
        try:
            selected_layers = self.get_selected_layers()
            self._update_grid_view(selected_layers)
            self._update_contour_controls_visibility()

            layer_name = self.get_primary_layer_name()
            if not layer_name:
                if self.plot_type == 'CONTOUR':
                    self._clear_contour_plot()
                    self.canvas_widget.figure.canvas.draw_idle()
                return

            if len(selected_layers) > 1:
                common_harmonics = self._get_common_harmonics(selected_layers)
                if common_harmonics is not None and len(common_harmonics) > 0:
                    min_harmonic = int(np.min(common_harmonics))
                    max_harmonic = int(np.max(common_harmonics))
                    self.harmonic_spinbox.setRange(min_harmonic, max_harmonic)

            if self._user_axes_limits is None and self.has_phasor_data():
                ax = self.canvas_widget.axes
                self._user_axes_limits = (ax.get_xlim(), ax.get_ylim())

            self.plot()

        finally:
            self._in_on_selection_changed = False
            self._preserve_plot_type_on_restore = False

    def _update_grid_view(self, selected_layers):
        """Update napari grid view based on selected layers.

        When multiple layers are selected, enables grid mode and makes
        selected layers visible. When only one layer is selected,
        disables grid mode.

        Parameters
        ----------
        selected_layers : list of napari.layers.Image
            List of currently selected layers.
        """
        selected_names = {layer.name for layer in selected_layers}

        if len(selected_layers) > 1:
            self.viewer.grid.enabled = True

            for layer in self.viewer.layers:
                if (
                    isinstance(layer, Image)
                    and "G" in layer.metadata
                    and "S" in layer.metadata
                    and "G_original" in layer.metadata
                    and "S_original" in layer.metadata
                ):
                    layer.visible = layer.name in selected_names
        else:
            self.viewer.grid.enabled = False

            if selected_layers:
                selected_layers[0].visible = True

    def _get_common_harmonics(self, layers):
        """Get the intersection of harmonics available in all layers.

        Parameters
        ----------
        layers : list of napari.layers.Image
            List of image layers to check.

        Returns
        -------
        np.ndarray or None
            Array of common harmonic values, or None if no common harmonics.
        """
        if not layers:
            return None

        common = None
        for layer in layers:
            harmonics = layer.metadata.get("harmonics")
            if harmonics is None:
                continue
            harmonics = np.atleast_1d(harmonics)
            if common is None:
                common = set(harmonics.tolist())
            else:
                common = common.intersection(set(harmonics.tolist()))

        if common is None or len(common) == 0:
            return None
        return np.array(sorted(common))

    def _restore_original_phasor_data(self, image_layer):
        """Restore original G, S, and image data from backups.

        Parameters
        ----------
        image_layer : napari.layers.Image
            The image layer to restore data for.
        """
        image_layer.metadata['G'] = image_layer.metadata['G_original']
        image_layer.metadata['S'] = image_layer.metadata['S_original']
        image_layer.data = image_layer.metadata["original_mean"].copy()

    def _apply_mask_to_phasor_data(self, mask_layer, image_layer):
        """Apply mask to phasor data by setting G and S values outside mask to NaN.

        Parameters
        ----------
        mask_layer : napari.layers.Labels or napari.layers.Shapes
            The mask layer to apply.
        image_layer : napari.layers.Image
            The image layer to store mask metadata.
        """
        if isinstance(mask_layer, Shapes) and len(mask_layer.data) > 0:
            mask_data = mask_layer.to_labels(
                labels_shape=image_layer.data.shape
            )
        elif isinstance(mask_layer, Labels) and mask_layer.data.any():
            mask_data = mask_layer.data
        else:
            return

        image_layer.metadata['mask'] = mask_data.copy()

        mask_invalid = mask_data <= 0
        image_layer.data = np.where(mask_invalid, np.nan, image_layer.data)

        g_array = image_layer.metadata['G']
        s_array = image_layer.metadata['S']

        if g_array.ndim > image_layer.data.ndim:
            mask_invalid_expanded = mask_invalid[np.newaxis, ...]
            image_layer.metadata['G'] = np.where(
                mask_invalid_expanded, np.nan, g_array
            )
            image_layer.metadata['S'] = np.where(
                mask_invalid_expanded, np.nan, s_array
            )
        else:
            image_layer.metadata['G'] = np.where(mask_invalid, np.nan, g_array)
            image_layer.metadata['S'] = np.where(mask_invalid, np.nan, s_array)

    def _on_mask_layer_changed(self, text):
        """Handle changes to the mask layer combo box (single-layer mode)."""
        selected_layers = self.get_selected_layers()
        if not selected_layers:
            return

        # In single-layer mode, apply the same mask to all selected layers
        # and update the assignments dict accordingly
        for image_layer in selected_layers:
            self._restore_original_phasor_data(image_layer)

            if text == "None":
                if 'mask' in image_layer.metadata:
                    del image_layer.metadata['mask']
                self._mask_assignments.pop(image_layer.name, None)
            else:
                mask_layer = self.viewer.layers[text]
                self._apply_mask_to_phasor_data(mask_layer, image_layer)
                self._mask_assignments[image_layer.name] = text

        if hasattr(self, 'filter_tab'):
            self.filter_tab._on_image_layer_changed()
            first_layer = selected_layers[0]
            if (
                first_layer.metadata['settings'].get('filter', None)
                is not None
                and first_layer.metadata['settings'].get('threshold', None)
                is not None
            ):
                self.filter_tab.apply_button_clicked()

        self.plot()

    def _on_mask_data_changed(self, event):
        """Handle changes to the mask layer data."""
        mask_name = event.source.name

        # Check if this mask is relevant: either the combobox selection
        # (single-layer mode) or any per-layer assignment (multi-layer mode)
        selected_layers = self.get_selected_layers()
        if not selected_layers:
            return

        if len(selected_layers) <= 1:
            # Single-layer mode: use combobox
            if self.mask_layer_combobox.currentText() != mask_name:
                return
            affected_layers = selected_layers
        else:
            # Multi-layer mode: only affect layers assigned to this mask
            affected_layers = [
                layer
                for layer in selected_layers
                if self._mask_assignments.get(layer.name) == mask_name
            ]
            if not affected_layers:
                return

        mask_layer = event.source

        for image_layer in affected_layers:
            self._restore_original_phasor_data(image_layer)
            self._apply_mask_to_phasor_data(mask_layer, image_layer)

        if hasattr(self, 'filter_tab'):
            self.filter_tab._on_image_layer_changed()
            first_layer = selected_layers[0]
            if (
                first_layer.metadata['settings'].get('filter', None)
                is not None
                and first_layer.metadata['settings'].get('threshold', None)
                is not None
            ):
                self.filter_tab.apply_button_clicked()

        self.refresh_current_plot()

    def _update_mask_ui_mode(self):
        """Switch between single combobox and assign-masks button based on selection count."""
        selected = self.get_selected_layer_names()
        if len(selected) > 1:
            # Multi-layer mode: show button, hide combobox
            self.mask_layer_label.setVisible(False)
            self.mask_layer_combobox.setVisible(False)
            self.mask_assign_button.setVisible(True)
            self._update_mask_assign_button_text()
        else:
            # Single-layer mode: show combobox, hide button
            self.mask_layer_label.setVisible(True)
            self.mask_layer_combobox.setVisible(True)
            self.mask_assign_button.setVisible(False)
            # Sync the combobox with current assignment for the single selected layer
            if selected:
                current_mask = self._mask_assignments.get(selected[0], "None")
                self.mask_layer_combobox.blockSignals(True)
                self.mask_layer_combobox.setCurrentText(current_mask)
                self.mask_layer_combobox.blockSignals(False)

    def _update_mask_assign_button_text(self):
        """Update the mask assign button text to show assignment summary."""
        selected = self.get_selected_layer_names()
        assigned = [
            name
            for name in selected
            if self._mask_assignments.get(name, "None") != "None"
        ]
        if assigned:
            self.mask_assign_button.setText(
                f"({len(assigned)}/{len(selected)} layers masked)"
            )
        else:
            self.mask_assign_button.setText("Assign Masks...")

    def _open_mask_assignment_dialog(self):
        """Open the mask assignment dialog for multi-layer mode."""
        selected_names = self.get_selected_layer_names()
        if not selected_names:
            return

        mask_layer_names = [
            layer.name
            for layer in self.viewer.layers
            if isinstance(layer, (Labels, Shapes))
        ]

        dialog = MaskAssignmentDialog(
            image_layer_names=selected_names,
            mask_layer_names=mask_layer_names,
            current_assignments=self._mask_assignments,
            parent=self,
        )

        if dialog.exec_() == QDialog.Accepted:
            new_assignments = dialog.get_assignments()
            self._apply_mask_assignments(new_assignments)

    def _apply_mask_assignments(self, assignments):
        """Apply per-layer mask assignments.

        Parameters
        ----------
        assignments : dict
            Mapping of image layer name -> mask layer name (or "None").
        """
        self._mask_assignments = {
            k: v for k, v in assignments.items() if v != "None"
        }

        selected_layers = self.get_selected_layers()
        for image_layer in selected_layers:
            self._restore_original_phasor_data(image_layer)
            mask_name = assignments.get(image_layer.name, "None")

            if mask_name == "None":
                if 'mask' in image_layer.metadata:
                    del image_layer.metadata['mask']
            else:
                mask_layer = self.viewer.layers[mask_name]
                self._apply_mask_to_phasor_data(mask_layer, image_layer)

        if hasattr(self, 'filter_tab'):
            self.filter_tab._on_image_layer_changed()
            if selected_layers:
                first_layer = selected_layers[0]
                if (
                    first_layer.metadata['settings'].get('filter', None)
                    is not None
                    and first_layer.metadata['settings'].get('threshold', None)
                    is not None
                ):
                    self.filter_tab.apply_button_clicked()

        self._update_mask_assign_button_text()
        self.plot()

    def get_mask_for_layer(self, layer_name):
        """Get the mask layer name assigned to a specific image layer.

        Parameters
        ----------
        layer_name : str
            Name of the image layer.

        Returns
        -------
        str
            Name of the assigned mask layer, or "None".
        """
        selected = self.get_selected_layer_names()
        if len(selected) <= 1:
            return self.mask_layer_combobox.currentText()
        return self._mask_assignments.get(layer_name, "None")

    def refresh_phasor_data(self):
        """Reload phasor data from the current layer metadata and replot."""
        layer_name = (
            self.image_layer_with_phasor_features_combobox.currentText()
        )
        if not layer_name or layer_name not in self.viewer.layers:
            return

        layer = self.viewer.layers[layer_name]
        layer_metadata = layer.metadata

        self._g_array = layer_metadata.get("G")
        self._s_array = layer_metadata.get("S")
        self._g_original_array = layer_metadata.get("G_original")
        self._s_original_array = layer_metadata.get("S_original")
        self._harmonics_array = layer_metadata.get("harmonics")

        if self._harmonics_array is not None:
            self._harmonics_array = np.atleast_1d(self._harmonics_array)

        self.plot()

    def has_phasor_data(self):
        """Check if valid phasor data is loaded.

        Returns True if the primary layer has phasor data, or if any
        selected layer has valid G/S arrays.

        Returns
        -------
        bool
            True if phasor data is available, False otherwise.
        """
        if (
            self._g_array is not None
            and self._s_array is not None
            and self._harmonics_array is not None
        ):
            return True

        selected_layers = self.get_selected_layers()
        for layer in selected_layers:
            if (
                layer.metadata.get("G") is not None
                and layer.metadata.get("S") is not None
            ):
                return True

        return False

    def get_harmonic_index(self, harmonic=None):
        """Return index of harmonic in loaded harmonics array.

        Parameters
        ----------
        harmonic : int, optional
            Harmonic number to look up. Defaults to current widget harmonic.

        Returns
        -------
        int | None
            Index into the harmonic axis of G/S arrays, or None if not found.
        """
        if not self.has_phasor_data():
            return None

        target = self.harmonic if harmonic is None else int(harmonic)
        harmonics = np.atleast_1d(self._harmonics_array)
        try:
            return int(np.where(harmonics == target)[0][0])
        except Exception:  # noqa: BLE001
            return None

    def get_phasor_spatial_shape(self):
        """Return spatial (Y, X[, ...]) shape of loaded phasor arrays."""
        if not self.has_phasor_data():
            return None

        shape = self._g_array.shape
        if self._harmonics_array is not None:
            return shape[1:]
        return shape

    def get_masked_gs(
        self,
        harmonic=None,
        *,
        flat=False,
        return_valid_mask=False,
    ):
        """Get masked phasor G/S arrays for a harmonic.

        Masking convention: invalid pixels are represented as NaN in G/S.

        Parameters
        ----------
        harmonic : int, optional
            Harmonic number. Defaults to current widget harmonic.
        flat : bool
            If True, return 1D arrays (raveled). Otherwise return 2D arrays.
        return_valid_mask : bool
            If True, also return a boolean mask where both G and S are valid.

        Returns
        -------
        (g, s) or (g, s, valid)
            g and s are numpy arrays; valid is a boolean array of same shape.
            Returns (None, None[, None]) if data is unavailable.
        """
        if not self.has_phasor_data():
            if return_valid_mask:
                return None, None, None
            return None, None

        harmonic_idx = self.get_harmonic_index(harmonic)
        if harmonic_idx is None:
            if return_valid_mask:
                return None, None, None
            return None, None

        if self._harmonics_array is not None:
            g = self._g_array[harmonic_idx]
            s = self._s_array[harmonic_idx]
        else:
            g = self._g_array
            s = self._s_array

        valid = (~np.isnan(g)) & (~np.isnan(s))

        if flat:
            g = g.ravel()
            s = s.ravel()
            valid = valid.ravel()

        if return_valid_mask:
            return g, s, valid
        return g, s

    def get_features(self):
        """Get the G and S features for the selected harmonic.

        Merges data from all selected layers for combined plotting.

        Returns
        -------
        x_data : np.ndarray
            The G feature data (merged from all selected layers).
        y_data : np.ndarray
            The S feature data (merged from all selected layers).
        """
        return self.get_merged_features()

    def get_merged_features(self):
        """Get merged G and S features from all selected layers.

        Combines phasor data from all selected layers into single arrays
        for unified plotting. Each layer's data is extracted at the current
        harmonic and merged together.

        Returns
        -------
        tuple or None
            (g_merged, s_merged) arrays, or None if no valid data.
        """
        selected_layers = self.get_selected_layers()
        if not selected_layers:
            return None

        all_g = []
        all_s = []

        for layer in selected_layers:
            g_array = layer.metadata.get("G")
            s_array = layer.metadata.get("S")
            harmonics_array = layer.metadata.get("harmonics")

            if g_array is None or s_array is None:
                continue

            if harmonics_array is not None:
                harmonics_array = np.atleast_1d(harmonics_array)
                target_harmonic = self.harmonic
                try:
                    harmonic_idx = int(
                        np.where(harmonics_array == target_harmonic)[0][0]
                    )
                except (IndexError, ValueError):
                    continue
            else:
                harmonic_idx = 0

            if g_array.ndim > layer.data.ndim:
                g = g_array[harmonic_idx]
                s = s_array[harmonic_idx]
            else:
                g = g_array
                s = s_array

            g_flat = g.ravel()
            s_flat = s.ravel()
            valid = (~np.isnan(g_flat)) & (~np.isnan(s_flat))

            all_g.append(g_flat[valid])
            all_s.append(s_flat[valid])

        if not all_g:
            return None

        g_merged = np.concatenate(all_g)
        s_merged = np.concatenate(all_s)

        if len(g_merged) == 0:
            return None

        return g_merged, s_merged

    def _on_scroll_zoom(self, event):
        """Zoom the phasor plot axes centered on the mouse cursor.

        Scrolling up zooms in; scrolling down zooms out.
        """
        ax = self.canvas_widget.axes
        if event.inaxes is not ax:
            return

        ZOOM_FACTOR = 1.15  # zoom step (>1 means 15% range change per notch)
        scale = 1.0 / ZOOM_FACTOR if event.step > 0 else ZOOM_FACTOR

        x_mouse, y_mouse = event.xdata, event.ydata
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()

        # Push current view to toolbar history before zooming
        # This allows the Home button to restore the original view
        if (
            hasattr(self.canvas_widget, 'toolbar')
            and self.canvas_widget.toolbar
        ):
            self.canvas_widget.toolbar.push_current()

        # New limits centered on mouse position
        new_x_min = x_mouse - (x_mouse - x_min) * scale
        new_x_max = x_mouse + (x_max - x_mouse) * scale
        new_y_min = y_mouse - (y_mouse - y_min) * scale
        new_y_max = y_mouse + (y_max - y_mouse) * scale

        ax.set_xlim(new_x_min, new_x_max)
        ax.set_ylim(new_y_min, new_y_max)
        self._user_axes_limits = (ax.get_xlim(), ax.get_ylim())
        self.canvas_widget.canvas.draw_idle()

    def _patch_toolbar_limits(self):
        """Patch toolbar release_zoom, release_pan, and home to manage user limits."""
        toolbar = getattr(self.canvas_widget, 'toolbar', None)
        if toolbar is None:
            return

        plotter = self
        _original_release_zoom = getattr(toolbar, 'release_zoom', None)
        _original_release_pan = getattr(toolbar, 'release_pan', None)
        _original_home = getattr(toolbar, 'home', None)

        if _original_release_zoom is not None:

            def _release_zoom_patched(event):
                _original_release_zoom(event)
                ax = plotter.canvas_widget.axes
                plotter._user_axes_limits = (ax.get_xlim(), ax.get_ylim())

            toolbar.release_zoom = _release_zoom_patched

        if _original_release_pan is not None:

            def _release_pan_patched(event):
                _original_release_pan(event)
                ax = plotter.canvas_widget.axes
                plotter._user_axes_limits = (ax.get_xlim(), ax.get_ylim())

            toolbar.release_pan = _release_pan_patched

        if _original_home is not None:

            def _home_patched(*args, **kwargs):
                # Clear user-defined limits so _redefine_axes_limits recomputes
                # the correct default for the current mode (semi-circle vs full circle)
                plotter._user_axes_limits = None
                _original_home(*args, **kwargs)
                # Re-apply the correct default limits for the current mode
                plotter._redefine_axes_limits()

            toolbar.home = _home_patched

    def _patch_toolbar_save(self):
        """Monkey-patch figure.savefig to use black colors for export.

        Colors are swapped to black immediately before the file is written
        and restored right after, so the Qt event loop cannot reset them
        while the file-save dialog is open.
        """
        fig = self.canvas_widget.figure
        _original_savefig = fig.savefig
        plotter = self  # prevent closure over 'self' name collisions

        def _savefig_with_black(*args, **kwargs):
            saved = plotter._capture_plot_colors()
            plotter._apply_plot_colors("black")
            try:
                _original_savefig(*args, **kwargs)
            finally:
                plotter._apply_plot_colors_from_saved(saved)

        fig.savefig = _savefig_with_black

    def _capture_plot_colors(self):
        """Snapshot current colors of all plot elements.

        Returns a dict that can be passed to ``_apply_plot_colors_from_saved``.
        """
        saved = {"axes": [], "colorbar": None}

        for key in ("HISTOGRAM2D", "SCATTER"):
            artist = self.canvas_widget.artists.get(key)
            if artist is None:
                continue
            ax = artist.ax
            state = {
                "ax": ax,
                "xlabel_color": ax.xaxis.label.get_color(),
                "ylabel_color": ax.yaxis.label.get_color(),
                "title_color": ax.title.get_color(),
                "spine_colors": {
                    name: sp.get_edgecolor() for name, sp in ax.spines.items()
                },
                "xticklabel_colors": [
                    t.get_color() for t in ax.get_xticklabels()
                ],
                "yticklabel_colors": [
                    t.get_color() for t in ax.get_yticklabels()
                ],
                "xtick_colors": [
                    t.get_markeredgecolor() for t in ax.xaxis.get_ticklines()
                ],
                "ytick_colors": [
                    t.get_markeredgecolor() for t in ax.yaxis.get_ticklines()
                ],
            }
            saved["axes"].append(state)

        if self.colorbar is not None:
            cb_ax = self.colorbar.ax
            saved["colorbar"] = {
                "ylabel_color": cb_ax.yaxis.label.get_color(),
                "outline_color": self.colorbar.outline.get_edgecolor(),
                "ticklabel_colors": [
                    t.get_color() for t in cb_ax.get_yticklabels()
                ],
                "tick_colors": [
                    t.get_markeredgecolor()
                    for t in cb_ax.yaxis.get_ticklines()
                ],
            }

        saved["semi_circle"] = []
        for artist in self.semi_circle_plot_artist_list:
            if isinstance(artist, Patch):
                saved["semi_circle"].append(
                    ("edgecolor", artist.get_edgecolor())
                )
            else:
                saved["semi_circle"].append(("color", artist.get_color()))

        saved["polar"] = []
        for artist in self.polar_plot_artist_list:
            if isinstance(artist, Patch):
                saved["polar"].append(("edgecolor", artist.get_edgecolor()))
            else:
                saved["polar"].append(("color", artist.get_color()))

        return saved

    def _apply_plot_colors(self, color):
        """Set spines, labels, ticks, and colorbar elements to *color*."""
        for key in ("HISTOGRAM2D", "SCATTER"):
            artist = self.canvas_widget.artists.get(key)
            if artist is None:
                continue
            ax = artist.ax
            ax.xaxis.label.set_color(color)
            ax.yaxis.label.set_color(color)
            ax.title.set_color(color)
            ax.tick_params(colors=color, which="both")
            for sp in ax.spines.values():
                sp.set_edgecolor(color)

        if self.colorbar is not None:
            self.set_colorbar_style(color=color)

        if not self.white_background:
            for artist in self.semi_circle_plot_artist_list:
                if isinstance(artist, Patch):
                    artist.set_edgecolor(color)
                else:
                    artist.set_color(color)
            for artist in self.polar_plot_artist_list:
                if isinstance(artist, Patch):
                    artist.set_edgecolor(color)
                else:
                    artist.set_color(color)

    def _apply_plot_colors_from_saved(self, saved):
        """Restore plot element colors from a snapshot dict."""
        for state in saved["axes"]:
            ax = state["ax"]
            ax.xaxis.label.set_color(state["xlabel_color"])
            ax.yaxis.label.set_color(state["ylabel_color"])
            ax.title.set_color(state["title_color"])
            for name, sp in ax.spines.items():
                sp.set_edgecolor(state["spine_colors"][name])
            for t, c in zip(
                ax.get_xticklabels(), state["xticklabel_colors"], strict=False
            ):
                t.set_color(c)
            for t, c in zip(
                ax.get_yticklabels(), state["yticklabel_colors"], strict=False
            ):
                t.set_color(c)
            for t, c in zip(
                ax.xaxis.get_ticklines(), state["xtick_colors"], strict=False
            ):
                t.set_markeredgecolor(c)
            for t, c in zip(
                ax.yaxis.get_ticklines(), state["ytick_colors"], strict=False
            ):
                t.set_markeredgecolor(c)

        if saved["colorbar"] is not None and self.colorbar is not None:
            cb = saved["colorbar"]
            self.colorbar.ax.yaxis.label.set_color(cb["ylabel_color"])
            self.colorbar.outline.set_edgecolor(cb["outline_color"])
            for t, c in zip(
                self.colorbar.ax.get_yticklabels(),
                cb["ticklabel_colors"],
                strict=False,
            ):
                t.set_color(c)
            for t, c in zip(
                self.colorbar.ax.yaxis.get_ticklines(),
                cb["tick_colors"],
                strict=False,
            ):
                t.set_markeredgecolor(c)

        for artist, (attr_type, saved_color) in zip(
            self.semi_circle_plot_artist_list,
            saved.get("semi_circle", []),
            strict=False,
        ):
            if attr_type == "edgecolor":
                artist.set_edgecolor(saved_color)
            else:
                artist.set_color(saved_color)

        for artist, (attr_type, saved_color) in zip(
            self.polar_plot_artist_list, saved.get("polar", []), strict=False
        ):
            if attr_type == "edgecolor":
                artist.set_edgecolor(saved_color)
            else:
                artist.set_color(saved_color)

        self.canvas_widget.figure.canvas.draw_idle()

    def set_axes_labels(self):
        """Set the axes labels in the canvas widget."""
        text_color = "white"

        self.canvas_widget.artists['SCATTER'].ax.set_xlabel(
            "G", color=text_color, fontweight='bold'
        )
        self.canvas_widget.artists['SCATTER'].ax.set_ylabel(
            "S", color=text_color, fontweight='bold'
        )
        self.canvas_widget.artists['HISTOGRAM2D'].ax.set_xlabel(
            "G", color=text_color, fontweight='bold'
        )
        self.canvas_widget.artists['HISTOGRAM2D'].ax.set_ylabel(
            "S", color=text_color, fontweight='bold'
        )

        self.canvas_widget.artists['SCATTER'].ax.tick_params(
            colors=text_color, which='both'
        )
        self.canvas_widget.artists['HISTOGRAM2D'].ax.tick_params(
            colors=text_color, which='both'
        )

        for spine in self.canvas_widget.artists['SCATTER'].ax.spines.values():
            spine.set_color(text_color)
        for spine in self.canvas_widget.artists[
            'HISTOGRAM2D'
        ].ax.spines.values():
            spine.set_color(text_color)

    def _update_scatter_plot(self, x_data, y_data, selection_id_data=None):
        """Update the scatter plot with new data."""
        if len(x_data) == 0 or len(y_data) == 0:
            return

        plot_data = np.column_stack((x_data, y_data))
        self.canvas_widget.artists['SCATTER'].data = plot_data

        # Setting data causes biaplotter to reset size and alpha to default values
        # Re-apply the user's chosen size and alpha
        self.canvas_widget.artists['SCATTER'].size = (
            self.plotter_inputs_widget.marker_size_spinbox.value()
        )
        self.canvas_widget.artists['SCATTER'].alpha = (
            self.plotter_inputs_widget.marker_alpha_spinbox.value()
        )

        # Only update color_indices if changed
        target_color_indices = (
            selection_id_data if selection_id_data is not None else 0
        )
        if isinstance(target_color_indices, np.ndarray):
            target_color_indices = target_color_indices.astype(int)
        else:
            target_color_indices = int(target_color_indices)

        # For arrays, use array_equal; for scalars, use direct comparison
        indices_changed = False
        if isinstance(target_color_indices, np.ndarray):
            if (
                self._last_scatter_color_indices is None
                or not isinstance(self._last_scatter_color_indices, np.ndarray)
                or not np.array_equal(
                    self._last_scatter_color_indices, target_color_indices
                )
            ):
                indices_changed = True
        else:
            # target_color_indices is a scalar
            if (
                self._last_scatter_color_indices is None
                or isinstance(self._last_scatter_color_indices, np.ndarray)
                or self._last_scatter_color_indices != target_color_indices
            ):
                indices_changed = True

        if indices_changed:
            self.canvas_widget.artists['SCATTER'].color_indices = (
                target_color_indices
            )
            self._last_scatter_color_indices = target_color_indices

    def _update_histogram_plot(self, x_data, y_data, selection_id_data=None):
        """Update the histogram plot with new data."""
        plot_data = np.column_stack((x_data, y_data))
        histogram_artist = self.canvas_widget.artists['HISTOGRAM2D']

        histogram_artist.data = plot_data
        histogram_artist.cmin = 1

        if self._last_histogram_bins != self.histogram_bins:
            histogram_artist.bins = self.histogram_bins
            self._last_histogram_bins = self.histogram_bins

        histogram_is_solid = self._histogram_style == 'solid'
        if histogram_is_solid:
            cache_key = f"solid:{tuple(self._histogram_color)}"
        else:
            cache_key = self.histogram_colormap

        if self._last_histogram_colormap != cache_key:
            if histogram_is_solid:
                selected_histogram_colormap = self._make_solid_contour_cmap(
                    "histogram_solid",
                    self._histogram_color,
                )
            else:
                selected_histogram_colormap = colormaps.ALL_COLORMAPS[
                    self.histogram_colormap
                ]
                selected_histogram_colormap = (
                    LinearSegmentedColormap.from_list(
                        self.histogram_colormap,
                        selected_histogram_colormap.colors,
                    )
                )
            histogram_artist.histogram_colormap = selected_histogram_colormap
            self._last_histogram_colormap = cache_key
            self._last_histogram_colormap_object = selected_histogram_colormap
        else:
            selected_histogram_colormap = self._last_histogram_colormap_object

        if histogram_artist.histogram is not None:
            current_norm = "log" if self.histogram_log_scale else "linear"
            if self._last_histogram_norm != current_norm:
                histogram_artist.histogram_color_normalization_method = (
                    current_norm
                )
                self._last_histogram_norm = current_norm

        target_color_indices = (
            selection_id_data if selection_id_data is not None else 0
        )

        indices_changed = False
        if isinstance(target_color_indices, np.ndarray):
            if (
                self._last_histogram_color_indices is None
                or not isinstance(
                    self._last_histogram_color_indices, np.ndarray
                )
                or not np.array_equal(
                    self._last_histogram_color_indices, target_color_indices
                )
            ):
                indices_changed = True
        else:
            if (
                self._last_histogram_color_indices is None
                or isinstance(self._last_histogram_color_indices, np.ndarray)
                or self._last_histogram_color_indices != target_color_indices
            ):
                indices_changed = True

        if indices_changed:
            histogram_artist.color_indices = target_color_indices
            self._last_histogram_color_indices = target_color_indices

        self._update_colorbar(selected_histogram_colormap)

    def _clear_contour_plot(self):
        """Clear all contour collections from the plot."""
        # Clear tracked collections
        for c in getattr(self, '_contour_collections', []):
            with contextlib.suppress(Exception):
                c.remove()
                # Fallback for older Matplotlib versions if remove fails
            with contextlib.suppress(Exception):
                if hasattr(c, 'collections'):
                    for col in c.collections:
                        col.remove()
        self._contour_collections = []

        # Cleanup ANY lingering contour elements in the axes using labels
        ax = self.canvas_widget.axes
        for artist in list(ax.collections):
            if artist.get_label() == 'contour_plot_element':
                with contextlib.suppress(Exception):
                    artist.remove()

        legend = ax.get_legend()
        if legend is not None:
            with contextlib.suppress(Exception):
                legend.remove()

        self._remove_colorbar()

    def _resolve_contour_colormap(self):
        """Resolve selected colormap to a Matplotlib colormap object."""
        cmap_name = self.plotter_inputs_widget.colormap_combobox.currentText()
        if self._has_multiple_selected_layers():
            cmap_name = (
                self._contour_multi_layer_colormap
                or self.plotter_inputs_widget.colormap_combobox.currentText()
            )
            if cmap_name == "Select color...":
                # Sentinel entry for solid style; keep a valid fallback colormap.
                cmap_name = "turbo"
        elif cmap_name == "Select color...":
            cmap_name = self._single_contour_colormap or "turbo"

        return resolve_colormap_by_name(cmap_name)

    def _get_selected_layer_feature_map(self):
        """Return valid flattened G/S data for each selected layer."""
        selected_layers = self.get_selected_layers()
        if not selected_layers:
            return {}

        data_by_layer = {}
        target_harmonic = self.harmonic

        for layer in selected_layers:
            g_array = layer.metadata.get("G")
            s_array = layer.metadata.get("S")
            harmonics_array = layer.metadata.get("harmonics")

            if g_array is None or s_array is None:
                continue

            if harmonics_array is not None:
                harmonics_array = np.atleast_1d(harmonics_array)
                try:
                    harmonic_idx = int(
                        np.where(harmonics_array == target_harmonic)[0][0]
                    )
                except (IndexError, ValueError):
                    continue
            else:
                harmonic_idx = 0

            if g_array.ndim > layer.data.ndim:
                g = g_array[harmonic_idx]
                s = s_array[harmonic_idx]
            else:
                g = g_array
                s = s_array

            g_flat = g.ravel()
            s_flat = s.ravel()
            valid = (~np.isnan(g_flat)) & (~np.isnan(s_flat))
            if not np.any(valid):
                continue

            data_by_layer[layer.name] = (g_flat[valid], s_flat[valid])

        return data_by_layer

    def _compute_contour_histogram(
        self, x_data, y_data, bins, range_xlim, range_ylim
    ):
        h, xedges, yedges = np.histogram2d(
            x_data, y_data, bins=bins, range=[range_xlim, range_ylim]
        )
        xcenters = xedges[:-1] + ((xedges[1] - xedges[0]) / 2.0)
        ycenters = yedges[:-1] + ((yedges[1] - yedges[0]) / 2.0)

        # Keep histogram in count-space; log handling is applied at contour() via norm='log'.
        h = h.astype(float)
        h[h <= 0] = np.nan

        return h, xcenters, ycenters

    @staticmethod
    def _normalize_rgb(color):
        arr = np.asarray(color, dtype=float)
        if arr.max(initial=0) > 1.0:
            arr = arr / 255.0
        return tuple(arr[:3])

    def _make_solid_contour_cmap(self, name, target_color):
        """Create a subtle ramp for solid contour style.

        Uses an off-white start so the lowest contour is still visible on
        white backgrounds while preserving the selected hue progression.
        """
        target = np.asarray(self._normalize_rgb(target_color), dtype=float)

        # Slightly below white with a tiny tint toward the target color.
        # This keeps low levels visible on white backgrounds.
        low_gray = 0.92
        tint = 0.08
        low_color = np.clip((1.0 - tint) * low_gray + tint * target, 0.0, 1.0)

        return LinearSegmentedColormap.from_list(
            name,
            [tuple(low_color), tuple(target)],
        )

    def _sample_colors_from_cmap(self, cmap, n_colors):
        if n_colors <= 1:
            return [cmap(0.6)]
        return [cmap(i / max(n_colors - 1, 1)) for i in range(n_colors)]

    def _update_contour_plot(self, x_data, y_data, selection_id_data=None):
        """Update or create the contour plot."""
        ax = self.canvas_widget.axes
        self._clear_contour_plot()

        levels = self.plotter_inputs_widget.contour_levels_spinbox.value()
        linewidths = (
            self.plotter_inputs_widget.contour_linewidth_spinbox.value()
        )
        use_log_norm = True
        cmap = self._resolve_contour_colormap()
        if cmap is None:
            return

        bins = self.plotter_inputs_widget.number_of_bins_spinbox.value()

        range_xlim = ax.get_xlim()
        range_ylim = ax.get_ylim()

        # Calculate aspect maintaining bins similar to histogram
        aspect = (range_xlim[1] - range_xlim[0]) / (
            range_ylim[1] - range_ylim[0]
        )
        if aspect > 1:
            bins = (bins, max(int(bins / aspect), 1))
        else:
            bins = (max(int(bins * aspect), 1), bins)

        layer_data = self._get_selected_layer_feature_map()
        has_multiple_layers = len(layer_data) > 1
        display_mode = (
            self._contour_display_mode if has_multiple_layers else "Merged"
        )

        def _tag_contour_set(cs_obj, label):
            with contextlib.suppress(Exception):
                if hasattr(cs_obj, 'collections'):
                    for col in cs_obj.collections:
                        col.set_label('contour_plot_element')
                else:
                    cs_obj.set_label('contour_plot_element')
                if label:
                    cs_obj.collections[0].set_label(label)

        if display_mode == "Merged" or not has_multiple_layers:
            h, xedges, yedges = self._compute_contour_histogram(
                x_data, y_data, bins, range_xlim, range_ylim
            )

            merged_cmap = cmap
            if has_multiple_layers:
                if self._contour_merged_style == 'solid':
                    merged_cmap = self._make_solid_contour_cmap(
                        "merged_solid",
                        self._contour_merged_color,
                    )
                else:
                    resolved = resolve_colormap_by_name(
                        self._contour_multi_layer_colormap
                    )
                    if resolved is not None:
                        merged_cmap = resolved
            else:
                if self._single_contour_style == 'solid':
                    merged_cmap = self._make_solid_contour_cmap(
                        "single_solid",
                        self._single_contour_color,
                    )
                else:
                    single_name = (
                        self._single_contour_colormap
                        or self.plotter_inputs_widget.colormap_combobox.currentText()
                    )
                    resolved = resolve_colormap_by_name(single_name)
                    if resolved is not None:
                        merged_cmap = resolved

            cs = ax.contour(
                xedges,
                yedges,
                h.T,
                levels=levels,
                linewidths=linewidths,
                cmap=merged_cmap,
                norm='log' if use_log_norm else None,
            )
            _tag_contour_set(cs, None)
            self._contour_collections.append(cs)
            legend = ax.get_legend()
            if legend is not None:
                with contextlib.suppress(Exception):
                    legend.remove()
            return

        # No colorbar for multi-series rendering; use legend instead.
        self._remove_colorbar()

        if display_mode == "Individual layers":
            items = list(layer_data.items())
            default_colors = self._sample_colors_from_cmap(cmap, len(items))
            legend_handles = []
            legend_labels = []

            for idx, (name, (lx, ly)) in enumerate(items):
                h, xedges, yedges = self._compute_contour_histogram(
                    lx, ly, bins, range_xlim, range_ylim
                )

                style = self._contour_layer_styles.get(name, {})
                style_mode = style.get("mode")
                if style_mode not in ("colormap", "solid"):
                    style_mode = "colormap"

                if style_mode == "colormap":
                    style_cmap_name = style.get(
                        "colormap", self._contour_multi_layer_colormap
                    )
                    style_cmap = resolve_colormap_by_name(style_cmap_name)
                    if style_cmap is None:
                        style_cmap = cmap
                    cs = ax.contour(
                        xedges,
                        yedges,
                        h.T,
                        levels=levels,
                        linewidths=linewidths,
                        cmap=style_cmap,
                        norm='log' if use_log_norm else None,
                    )
                    legend_handles.append(
                        ColormapLegendProxy(
                            style_cmap,
                            linewidths,
                            style="categorical",
                            n_colors=max(int(levels), 2),
                        )
                    )
                else:
                    custom = style.get(
                        "color", self._contour_layer_colors.get(name)
                    )
                    if custom is None:
                        custom = default_colors[idx]
                    color = self._normalize_rgb(custom)
                    solid_cmap = self._make_solid_contour_cmap(
                        f"solid_{name}", color
                    )
                    cs = ax.contour(
                        xedges,
                        yedges,
                        h.T,
                        levels=levels,
                        linewidths=linewidths,
                        cmap=solid_cmap,
                        norm='log' if use_log_norm else None,
                    )
                    legend_handles.append(
                        ColormapLegendProxy(
                            solid_cmap,
                            linewidths,
                            style="categorical",
                            n_colors=max(int(levels), 2),
                        )
                    )

                _tag_contour_set(cs, name)
                self._contour_collections.append(cs)
                legend_labels.append(name)

            if self._contour_show_legend and legend_handles:
                ax.legend(
                    handles=legend_handles,
                    labels=legend_labels,
                    loc='upper right',
                    frameon=False,
                    handler_map={ColormapLegendProxy: ColormapLegendHandler()},
                )
            return

        # Grouped mode
        grouped_data = {}
        for layer_name, (lx, ly) in layer_data.items():
            gid = int(self._contour_group_assignments.get(layer_name, 1))
            grouped_data.setdefault(gid, []).append((layer_name, lx, ly))

        group_ids = sorted(grouped_data)
        default_colors = self._sample_colors_from_cmap(cmap, len(group_ids))
        legend_handles = []
        legend_labels = []

        for idx, gid in enumerate(group_ids):
            group_label = self._contour_group_names.get(gid, f"Group {gid}")
            style = self._contour_group_styles.get(gid, {})
            style_mode = style.get("mode")
            if style_mode not in ("colormap", "solid"):
                style_mode = "colormap"

            style_cmap = None
            color = None
            if style_mode == "colormap":
                style_cmap_name = style.get(
                    "colormap", self._contour_multi_layer_colormap
                )
                style_cmap = resolve_colormap_by_name(style_cmap_name)
                if style_cmap is None:
                    style_cmap = cmap
                legend_handles.append(
                    ColormapLegendProxy(
                        style_cmap,
                        linewidths,
                        style="categorical",
                        n_colors=max(int(levels), 2),
                    )
                )
            else:
                custom = style.get(
                    "color", self._contour_group_colors.get(gid)
                )
                if custom is None:
                    custom = default_colors[idx]
                color = self._normalize_rgb(custom)
                solid_cmap = self._make_solid_contour_cmap(
                    f"solid_group_{gid}", color
                )
                legend_handles.append(
                    ColormapLegendProxy(
                        solid_cmap,
                        linewidths,
                        style="categorical",
                        n_colors=max(int(levels), 2),
                    )
                )

            for member_idx, (_layer_name, gx, gy) in enumerate(
                grouped_data[gid]
            ):
                h, xedges, yedges = self._compute_contour_histogram(
                    gx, gy, bins, range_xlim, range_ylim
                )
                if style_mode == "colormap":
                    cs = ax.contour(
                        xedges,
                        yedges,
                        h.T,
                        levels=levels,
                        linewidths=linewidths,
                        cmap=style_cmap,
                        norm='log' if use_log_norm else None,
                    )
                else:
                    cs = ax.contour(
                        xedges,
                        yedges,
                        h.T,
                        levels=levels,
                        linewidths=linewidths,
                        cmap=solid_cmap,
                        norm='log' if use_log_norm else None,
                    )

                _tag_contour_set(cs, group_label if member_idx == 0 else None)
                self._contour_collections.append(cs)

            legend_labels.append(group_label)

        if self._contour_show_legend and legend_handles:
            ax.legend(
                handles=legend_handles,
                labels=legend_labels,
                loc='upper right',
                frameon=False,
                handler_map={ColormapLegendProxy: ColormapLegendHandler()},
            )

    def _update_colorbar(self, colormap=None, mappable=None):
        """Update or create colorbar for the current plot."""
        self._remove_colorbar()

        # Determine which axes to use for inset
        ax = self.canvas_widget.axes
        self.cax = ax.inset_axes([1.05, 0, 0.05, 1])

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Attempt to set non-positive ylim on a log-scaled axis will be ignored*",
                category=UserWarning,
            )
            if mappable is not None:
                # Use a specific formatter that allows non-decade log labels for contour levels
                fmt = None

                is_log = False
                if (
                    getattr(mappable, 'norm', None) == 'log'
                    or getattr(mappable, 'norm', None)
                    and isinstance(mappable.norm, LogNorm)
                ):
                    is_log = True

                if is_log:
                    fmt = ticker.LogFormatterSciNotation(labelOnlyBase=False)
                    # If norm='log' was passed to contour, the norm's vmin/vmax might be None (or it might just be the string 'log'),
                    # which makes Colorbar draw an empty box. We set them explicitly from the evaluated levels on the mappable.
                    if getattr(mappable, 'norm', None) == 'log':
                        levels = getattr(mappable, 'levels', [])
                        valid_levels = [lv for lv in levels if lv > 0]
                        if valid_levels:
                            mappable.norm = LogNorm(
                                vmin=min(valid_levels), vmax=max(valid_levels)
                            )
                    elif (
                        getattr(mappable.norm, 'vmin', None) is None
                        or getattr(mappable.norm, 'vmax', None) is None
                    ):
                        levels = getattr(mappable, 'levels', [])
                        valid_levels = [lv for lv in levels if lv > 0]
                        if valid_levels:
                            mappable.norm.vmin = min(valid_levels)
                            mappable.norm.vmax = max(valid_levels)

                self.colorbar = Colorbar(
                    ax=self.cax, mappable=mappable, format=fmt
                )
            elif self.plot_type == 'HISTOGRAM2D' and colormap is not None:
                artist = self.canvas_widget.artists['HISTOGRAM2D']
                norm = artist._get_normalization(
                    artist.histogram[0], is_overlay=False
                )
                self.colorbar = Colorbar(ax=self.cax, cmap=colormap, norm=norm)

        if self.colorbar is not None:
            self.set_colorbar_style(
                color=(
                    "white"
                    if self.plotter_inputs_widget.white_background_checkbox.isChecked()
                    else "black"
                )
            )

    def _remove_colorbar(self):
        """Remove colorbar if it exists."""
        if self.colorbar is not None:
            with contextlib.suppress(Exception):
                self.colorbar.remove()
            self.colorbar = None
        if hasattr(self, 'cax') and self.cax is not None:
            with contextlib.suppress(Exception):
                self.cax.remove()
            self.cax = None

    def _update_plot_elements(self):
        """Update common plot elements like semicircle, axes, etc."""
        if self.toggle_semi_circle:
            self._update_semi_circle_plot(self.canvas_widget.axes)

        self._enforce_axes_aspect()
        self._update_plot_bg_color()

    def plot(self, x_data=None, y_data=None, selection_id_data=None):
        """Plot the selected phasor features efficiently."""
        if not self.has_phasor_data():
            if self.plot_type == 'CONTOUR':
                self._clear_contour_plot()
                self.canvas_widget.figure.canvas.draw_idle()
            return

        if getattr(self, '_updating_plot', False):
            return

        self._updating_plot = True

        try:
            if x_data is None or y_data is None:
                features = self.get_features()
                if features is None:
                    return
                x_data, y_data = features

            if len(x_data) == 0 or len(y_data) == 0:
                return

            self._set_active_artist_and_plot(
                self.plot_type, x_data, y_data, selection_id_data
            )

            if (
                hasattr(self, 'phasor_mapping_tab')
                and selection_id_data is None
                and self.tab_widget.currentWidget() is self.phasor_mapping_tab
            ):
                self.phasor_mapping_tab.reapply_if_active()

            # If the user has set a custom zoom, always restore it after
            # plotting — biaplotter resets the axes limits in its data setter.
            if self._user_axes_limits is not None:
                xlim, ylim = self._user_axes_limits
                self.canvas_widget.axes.set_xlim(xlim)
                self.canvas_widget.axes.set_ylim(ylim)
                self.canvas_widget.figure.canvas.draw_idle()
        finally:
            self._updating_plot = False

    def refresh_current_plot(self):
        """Refresh the current plot with existing data."""
        self.plot()

    def switch_plot_type(self, new_plot_type):
        """Switch between plot types efficiently."""
        self._connect_active_artist_signals()

        features = self.get_features()
        if features is not None:
            x_data, y_data = features
            self._set_active_artist_and_plot(
                new_plot_type, x_data, y_data, selection_id_data=None
            )

            if hasattr(
                self, 'selection_tab'
            ) and self.selection_tab.selection_id not in [None, "None", ""]:
                self.selection_tab.update_phasor_plot_with_selection_id(
                    self.selection_tab.selection_id
                )

    def _set_active_artist_and_plot(
        self, plot_type, x_data, y_data, selection_id_data=None
    ):
        """Set the active artist and update only the relevant plot."""
        if len(x_data) == 0 or len(y_data) == 0:
            return
        if plot_type != self.plot_type:
            self.plotter_inputs_widget.plot_type_combobox.blockSignals(True)
            self.plotter_inputs_widget.plot_type_combobox.setCurrentText(
                plot_type
            )
            self.plotter_inputs_widget.plot_type_combobox.blockSignals(False)
            self._connect_active_artist_signals()

        # Make sure biaplotter artists are hidden if switching to CONTOUR
        current_active = getattr(self.canvas_widget, 'active_artist', None)

        if plot_type == 'CONTOUR':
            for _name, artist in getattr(
                self.canvas_widget, 'artists', {}
            ).items():
                if hasattr(artist, 'visible'):
                    artist.visible = False
            self.canvas_widget.active_artist = None

        else:
            # Hide contour items when switching away
            self._clear_contour_plot()
            self.canvas_widget.figure.canvas.draw_idle()

        if plot_type == 'HISTOGRAM2D':
            self._update_histogram_plot(x_data, y_data, selection_id_data)
        elif plot_type == 'SCATTER':
            self._remove_colorbar()
            self._update_scatter_plot(x_data, y_data, selection_id_data)
        elif plot_type == 'CONTOUR':
            self._update_contour_plot(x_data, y_data, selection_id_data)

        if current_active != plot_type and plot_type in getattr(
            self.canvas_widget, 'artists', {}
        ):
            self.canvas_widget.active_artist = plot_type

        if (
            plot_type not in getattr(self.canvas_widget, 'artists', {})
            and plot_type != 'CONTOUR'
        ):
            return

        self._update_plot_elements()

    def set_colorbar_style(self, color="white"):
        """Set the colorbar style in the canvas widget."""
        # Color the ticks and their labels (both major and minor for log scale)
        self.colorbar.ax.yaxis.set_tick_params(
            color=color, labelcolor=color, which='both'
        )
        self.colorbar.ax.tick_params(axis='y', colors=color, which='both')

        # Color the bounding box (outline)
        self.colorbar.outline.set_edgecolor(color)

        # Ensure all spines are colored
        for spine in self.colorbar.ax.spines.values():
            spine.set_edgecolor(color)

        # Set the label with correct color
        label_text = (
            "Log10(Count)"
            if isinstance(self.colorbar.norm, LogNorm)
            else "Count"
        )
        self.colorbar.set_label(label_text, color=color)

        # Force update of tick labels (sometimes needed for older matplotlib)
        for tick_label in self.colorbar.ax.get_yticklabels():
            tick_label.set_color(color)

    def closeEvent(self, event):
        """Clean up signal connections and child widgets before closing."""
        self._is_closing = True

        # Stop background timers first.
        with contextlib.suppress(AttributeError):
            self._dock_check_timer.stop()
        with contextlib.suppress(AttributeError):
            self._analysis_dock_init_timer.stop()
        with contextlib.suppress(AttributeError):
            self._dock_resize_timer.stop()
        with contextlib.suppress(AttributeError):
            self._layer_selection_timer.stop()
        with contextlib.suppress(AttributeError):
            self._bins_timer.stop()

        # Disconnect timer callbacks to avoid queued invocations during teardown.
        with contextlib.suppress(TypeError, ValueError, AttributeError):
            self._dock_check_timer.timeout.disconnect(
                self._check_dock_visibility
            )
        with contextlib.suppress(TypeError, ValueError, AttributeError):
            self._analysis_dock_init_timer.timeout.disconnect(
                self._add_analysis_dock_widget
            )
        with contextlib.suppress(TypeError, ValueError, AttributeError):
            self._dock_resize_timer.timeout.disconnect(
                self._resize_initial_docks
            )
        with contextlib.suppress(TypeError, ValueError, AttributeError):
            self._layer_selection_timer.timeout.disconnect(
                self._process_layer_selection_change
            )
        with contextlib.suppress(TypeError, ValueError, AttributeError):
            self._bins_timer.timeout.disconnect(self._process_bins_change)

        # Disconnect viewer layer events owned by this widget.
        with contextlib.suppress(TypeError, ValueError, AttributeError):
            self.viewer.layers.events.inserted.disconnect(
                self.reset_layer_choices
            )
        with contextlib.suppress(TypeError, ValueError, AttributeError):
            self.viewer.layers.events.removed.disconnect(
                self.reset_layer_choices
            )

        # Disconnect combobox-driven callbacks.
        combo = getattr(self, 'image_layers_checkable_combobox', None)
        if combo is not None:
            with contextlib.suppress(TypeError, ValueError, AttributeError):
                combo.primaryLayerChanged.disconnect(
                    self._on_primary_layer_changed
                )
            with contextlib.suppress(TypeError, ValueError, AttributeError):
                combo.selectionChanged.disconnect(self._on_selection_changed)
            with contextlib.suppress(TypeError, ValueError, AttributeError):
                combo.selectionChanged.disconnect(self._update_mask_ui_mode)
            with contextlib.suppress(TypeError, ValueError, AttributeError):
                combo.primaryLayerChanged.disconnect(
                    self._sync_frequency_inputs_from_metadata
                )

        # Ensure child tabs run their own cleanup.
        for tab_name in (
            'calibration_tab',
            'filter_tab',
            'selection_tab',
            'components_tab',
            'phasor_mapping_tab',
            'fret_tab',
        ):
            tab = getattr(self, tab_name, None)
            if tab is not None:
                with contextlib.suppress(Exception):
                    tab.close()

        super().closeEvent(event)
