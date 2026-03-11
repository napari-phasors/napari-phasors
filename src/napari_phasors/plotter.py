import contextlib
import copy
import html
import math
import warnings
from pathlib import Path

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
    HistogramDockWidget,
    HistogramWidget,
    StatisticsDockWidget,
    update_frequency_in_metadata,
)
from .calibration_tab import CalibrationWidget
from .components_tab import ComponentsWidget
from .filter_tab import FilterWidget
from .fret_tab import FretWidget
from .phasor_mapping_tab import LifetimeWidget
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

        Note: This doesn't pass the text as argument like the original signal,
        but connected slots should handle being called without arguments.
        """
        return self._plotter.image_layers_checkable_combobox.selectionChanged


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
    lifetime_tab : QWidget
        The Lifetime tab for lifetime analysis.
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
        canvas_container = QWidget()
        canvas_container.setLayout(QVBoxLayout())
        canvas_container.layout().setContentsMargins(0, 0, 0, 0)
        canvas_container.layout().setSpacing(0)
        self.layout().addWidget(
            canvas_container, 1
        )  # stretch factor 1 to prioritize canvas

        # Load canvas widget (fixed at the top)
        self.canvas_widget = CanvasWidget(
            napari_viewer, highlight_enabled=False
        )
        self.canvas_widget.axes.set_aspect(1, adjustable='box')
        self.canvas_widget.setMinimumSize(300, 300)
        self.canvas_widget.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        self.set_axes_labels()
        canvas_container.layout().addWidget(self.canvas_widget)

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
        controls_container = QWidget()
        controls_container.setLayout(QVBoxLayout())
        controls_container.layout().setContentsMargins(10, 10, 10, 10)
        controls_container.layout().setSpacing(3)
        controls_container.setMaximumHeight(
            220
        )  # Prevent controls from growing too large
        self.layout().addWidget(
            controls_container, 0
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
        controls_container.layout().addWidget(image_layer_widget)

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

        controls_container.layout().addLayout(harmonics_and_mask_container)

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
        controls_container.layout().addWidget(import_buttons_widget)

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
        controls_container.layout().addWidget(self._dock_buttons_widget)

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
        self.analysis_widget.installEventFilter(self)

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

        # Canvas minimum is already set by canvas_widget.setMinimumSize(300, 300)
        # Controls container will size to its content

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
        self._create_lifetime_tab()
        self._create_fret_tab()

        # Connect napari signals when new layer is inseted or removed
        self.viewer.layers.events.inserted.connect(self.reset_layer_choices)
        self.viewer.layers.events.removed.connect(self.reset_layer_choices)

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
        self.import_from_layer_button.clicked.connect(
            self._import_settings_from_layer
        )
        self.import_from_file_button.clicked.connect(
            self._import_settings_from_file
        )

        # Populate plot type combobox
        self.plotter_inputs_widget.plot_type_combobox.addItems(
            ['HISTOGRAM2D', 'SCATTER']
        )

        # Populate colormap combobox
        self.plotter_inputs_widget.colormap_combobox.addItems(
            list(colormaps.ALL_COLORMAPS.keys())
        )
        self.histogram_colormap = "turbo"

        # Initialize attributes
        self.polar_plot_artist_list = []
        self.semi_circle_plot_artist_list = []
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

        The histogram container (QStackedWidget) is docked at the bottom.
        The analysis tab widget is then split to its right so they sit
        side-by-side, each taking roughly half the bottom width.
        """
        if (
            hasattr(self.viewer, 'window')
            and self.viewer.window is not None
            and hasattr(self.viewer.window, '_qt_window')
        ):
            qt_window = self.viewer.window._qt_window

            # Dock order: statistics (left) | histogram (middle) | analysis (right)

            # Statistics dock — leftmost
            self._statistics_dock = self.viewer.window.add_dock_widget(
                self.statistics_container,
                name="Statistics",
                area="bottom",
            )

            # Histogram dock — middle
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
            self._docks_initialized = True

            # Arrange side-by-side: statistics | histogram | analysis
            qt_window.splitDockWidget(
                self._statistics_dock,
                self._histogram_dock,
                Qt.Horizontal,
            )
            qt_window.splitDockWidget(
                self._histogram_dock,
                self._analysis_dock,
                Qt.Horizontal,
            )

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
                    self._statistics_dock,
                    self._histogram_dock,
                    self._analysis_dock,
                ],
                [300, 500, 500],
                Qt.Horizontal,
            )

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
            'number_of_bins': self.histogram_bins,
            'log_scale': self.histogram_log_scale,
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
            if 'plot_type' in settings:
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
            ("Lifetime", "lifetime_tab", "lifetime"),
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
                "lifetime_tab",
                "fret_tab",
                "components_tab",
                "selection_tab",
            ]

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
        if "lifetime_tab" in selected_tabs and hasattr(self, 'lifetime_tab'):
            self.lifetime_tab._on_image_layer_changed()
        if "fret_tab" in selected_tabs and hasattr(self, 'fret_tab'):
            self.fret_tab._on_image_layer_changed()
        if "selection_tab" in selected_tabs and hasattr(self, 'selection_tab'):
            self.selection_tab._on_image_layer_changed()

        current_tab_index = self.tab_widget.currentIndex()
        self._on_tab_changed(current_tab_index)

    def _apply_calibration_if_needed(self):
        """Apply calibration transformation if needed."""
        layer_name = (
            self.image_layer_with_phasor_features_combobox.currentText()
        )
        if not layer_name or layer_name not in self.viewer.layers:
            return
        layer = self.viewer.layers[layer_name]
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
                layer_name, phi_zero, mod_zero
            )
            layer.metadata["calibration_applied"] = True

    def _import_settings_from_layer(self):
        """Import all settings and analyses from another layer, with selection dialog."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Import Settings from Layer")
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("Select layer to import settings from:"))

        layer_combo = QComboBox()
        current_layer = (
            self.image_layer_with_phasor_features_combobox.currentText()
        )
        available_layers = [
            layer.name
            for layer in self.viewer.layers
            if isinstance(layer, Image)
            and all(
                key in layer.metadata
                for key in ["G", "S", "G_original", "S_original"]
            )
            and layer.name != current_layer
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
        """Copy metadata from source layer to current layer and apply selected analyses."""
        try:
            source_layer = self.viewer.layers[source_layer_name]
            current_layer_name = (
                self.image_layer_with_phasor_features_combobox.currentText()
            )
            if not current_layer_name:
                notifications.WarningNotification("No layer selected")
                return
            current_layer = self.viewer.layers[current_layer_name]
            import copy

            if "frequency" in selected_tabs:
                if (
                    'settings' in source_layer.metadata
                    and 'frequency' in source_layer.metadata['settings']
                ):
                    freq_val = source_layer.metadata['settings']['frequency']
                    update_frequency_in_metadata(current_layer, freq_val)
                    self._broadcast_frequency_value_across_tabs(str(freq_val))
                selected_tabs = [
                    tab for tab in selected_tabs if tab != "frequency"
                ]

            if 'settings' in source_layer.metadata:
                current_layer.metadata['settings'] = copy.deepcopy(
                    source_layer.metadata['settings']
                )

            self._restore_plot_settings_from_metadata()
            self._restore_all_tab_analyses(selected_tabs)
            self.plot()
            notifications.show_info(
                f"Settings and analyses imported from {source_layer_name}"
            )
        except Exception as e:  # noqa: BLE001
            notifications.WarningNotification(
                f"Failed to import settings: {str(e)}"
            )

    def _apply_imported_settings(self, settings, selected_tabs):
        current_layer_name = (
            self.image_layer_with_phasor_features_combobox.currentText()
        )
        if not current_layer_name:
            notifications.WarningNotification("No layer selected")
            return
        current_layer = self.viewer.layers[current_layer_name]
        if 'settings' not in current_layer.metadata:
            current_layer.metadata['settings'] = {}

        current_layer.metadata['settings'] = copy.deepcopy(settings)

        if 'frequency' in selected_tabs and 'frequency' in settings:
            update_frequency_in_metadata(current_layer, settings['frequency'])
            self._broadcast_frequency_value_across_tabs(
                str(settings['frequency'])
            )
            selected_tabs = [
                tab for tab in selected_tabs if tab != "frequency"
            ]

        self._restore_plot_settings_from_metadata()
        self._restore_all_tab_analyses(selected_tabs)
        self.plot()

    def eventFilter(self, obj, event):
        """Notify the active tab when the analysis widget is resized."""
        if obj is self.analysis_widget and event.type() == QEvent.Resize:
            self._notify_current_tab_width()
        return super().eventFilter(obj, event)

    def _notify_current_tab_width(self):
        """Call update_layout on the currently visible tab with the analysis widget width."""
        current_tab = self.tab_widget.currentWidget()
        if current_tab is not None and hasattr(current_tab, 'update_layout'):
            current_tab.update_layout(self.analysis_widget.width())

    def _on_tab_changed(self, index):
        """Handle tab change events to show/hide tab-specific lines."""
        current_tab = self.tab_widget.widget(index)

        if hasattr(self, 'lifetime_tab'):
            self.lifetime_tab.on_tab_visibility_changed(
                current_tab == self.lifetime_tab
            )

        self._hide_all_tab_artists()

        self._show_tab_artists(current_tab)

        # Show/hide histogram dock widgets based on active tab
        self._update_histogram_dock_visibility(current_tab)

        # Update filter histogram if switching to filter tab and it needs updating
        if hasattr(self, 'filter_tab') and current_tab == self.filter_tab:
            self.filter_tab._update_histogram_if_needed()

        self.canvas_widget.figure.canvas.draw_idle()

        # Notify the new tab about its current available width
        self._notify_current_tab_width()

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
        if current_tab == getattr(self, 'lifetime_tab', None):
            hist_idx = getattr(self, '_lifetime_hist_page_idx', 0)
            stats_idx = getattr(self, '_lifetime_stats_page_idx', 0)
            output_display = "Lifetime"
            with contextlib.suppress(AttributeError, RuntimeError):
                output_display = (
                    self.lifetime_tab.get_selected_output_display_name()
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
        """Make the statistics dock widget visible, re-adding it if it was closed."""
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

    def _show_histogram_dock(self):
        """Make the histogram dock widget visible, re-adding it if it was closed."""
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

        if not self._updating_settings:
            old_plot_type = getattr(self, '_current_plot_type', None)
            if new_plot_type != old_plot_type:
                self._current_plot_type = new_plot_type
                self._connect_active_artist_signals()
                self.switch_plot_type(new_plot_type)

    def _on_colormap_changed(self):
        """Callback for colormap change."""
        colormap = self.plotter_inputs_widget.colormap_combobox.currentText()
        self._update_setting_in_metadata('colormap', colormap)
        if not self._updating_settings and self.plot_type == 'HISTOGRAM2D':
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
        self.refresh_current_plot()

    def _on_log_scale_changed(self, state):
        """Callback for log scale change."""
        self._update_setting_in_metadata('log_scale', bool(state))
        if not self._updating_settings and self.plot_type == 'HISTOGRAM2D':
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Log normalization applied to color indices*",
                    category=UserWarning,
                )
                self.refresh_current_plot()

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
            self.plot()

    def on_white_background_changed(self):
        """Callback function when the white background checkbox is toggled."""
        self.set_axes_labels()
        if self.toggle_semi_circle:
            self._update_semi_circle_plot(self.canvas_widget.axes)
        else:
            self._update_polar_plot(self.canvas_widget.axes, visible=True)
        self.canvas_widget.figure.canvas.draw_idle()

        self.plot()

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
        self.plot()

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

    def _create_lifetime_tab(self):
        """Create the Lifetime tab."""
        self.lifetime_tab = LifetimeWidget(self.viewer, parent=self)
        self.tab_widget.addTab(self.lifetime_tab, "Phasor Mapping")

        # Wrap the histogram in a HistogramDockWidget and add to shared stack
        self.lifetime_histogram_dock_widget = HistogramDockWidget(
            self.lifetime_tab.histogram_widget,
            title="Output Histogram & Statistics",
        )

        self._lifetime_hist_page_idx = self._histogram_stack.addWidget(
            self.lifetime_histogram_dock_widget
        )

        # Create the matching statistics dock and link it
        self.lifetime_statistics_dock_widget = StatisticsDockWidget(
            self.lifetime_tab.histogram_widget,
            title="Output Statistics",
        )
        self._lifetime_stats_page_idx = self._statistics_stack.addWidget(
            self.lifetime_statistics_dock_widget
        )
        self.lifetime_histogram_dock_widget.link_statistics_dock(
            self.lifetime_statistics_dock_widget
        )
        self.lifetime_tab.frequency_input.editingFinished.connect(
            lambda: self._broadcast_frequency_value_across_tabs(
                self.lifetime_tab.frequency_input.text()
            )
        )
        self.lifetime_tab.outputTypeChanged.connect(
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
        self.lifetime_tab.frequency_input.blockSignals(True)
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
        self.lifetime_tab.frequency_input.setText(value)
        self.fret_tab.frequency_input.setText(value)

        self.calibration_tab.calibration_widget.frequency_input.blockSignals(
            False
        )
        self.lifetime_tab.frequency_input.blockSignals(False)
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
            circle_plot_limits = [0, 1, 0, 0.6]  # xmin, xmax, ymin, ymax
        else:
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
        return self.plotter_inputs_widget.plot_type_combobox.currentText()

    @plot_type.setter
    def plot_type(self, value):
        """Sets the plot type from the plot type combobox."""
        self.plotter_inputs_widget.plot_type_combobox.setCurrentText(value)

    @property
    def histogram_colormap(self):
        """Gets or sets the histogram colormap from the colormap combobox.

        Returns
        -------
        str
            The colormap name.
        """
        return self.plotter_inputs_widget.colormap_combobox.currentText()

    @histogram_colormap.setter
    def histogram_colormap(self, colormap: str):
        """Sets the histogram colormap from the colormap combobox."""
        if colormap not in colormaps.ALL_COLORMAPS:
            notifications.WarningNotification(
                f"{colormap} is not a valid colormap. Setting to default colormap."
            )
            colormap = self._histogram_colormap.name
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
        self.canvas_widget.active_artist.ax.set_aspect(1, adjustable='box')
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

            self._initialize_plot_settings_in_metadata(layer)

            self._restore_plot_settings_from_metadata()

            self._sync_frequency_inputs_from_metadata()

            if hasattr(self, 'filter_tab'):
                self.filter_tab._on_image_layer_changed()

            if hasattr(self, 'calibration_tab'):
                self.calibration_tab._on_image_layer_changed()

            if hasattr(self, 'selection_tab'):
                self.selection_tab._on_image_layer_changed()

            if hasattr(self, 'lifetime_tab'):
                self.lifetime_tab._on_image_layer_changed()

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

            self._initialize_plot_settings_in_metadata(layer)
            self._restore_plot_settings_from_metadata()

            if hasattr(self, 'filter_tab'):
                self.filter_tab._on_image_layer_changed()

            if hasattr(self, 'calibration_tab'):
                self.calibration_tab._on_image_layer_changed()

            if hasattr(self, 'selection_tab'):
                self.selection_tab._on_image_layer_changed()

            if hasattr(self, 'lifetime_tab'):
                self.lifetime_tab._on_image_layer_changed()

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
        self._layer_selection_timer.stop()
        self._layer_selection_timer.start()

    def _process_layer_selection_change(self):
        """Process layer selection change after debounce timer expires.

        Only updates the phasor plot. Tab UIs are not updated unless
        the primary layer changed (which triggers _on_primary_layer_changed).
        """
        if getattr(self, "_in_on_selection_changed", False):
            return
        self._in_on_selection_changed = True
        try:
            selected_layers = self.get_selected_layers()
            self._update_grid_view(selected_layers)

            layer_name = self.get_primary_layer_name()
            if not layer_name:
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

        if g_array.ndim == 3:
            mask_invalid_expanded = mask_invalid[np.newaxis, :, :]
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
        if self._g_array.ndim == 3:
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

        if self._g_array.ndim == 3:
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

            if g_array.ndim == 3:
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

        self.canvas_widget.artists['SCATTER'].ax.tick_params(colors=text_color)
        self.canvas_widget.artists['HISTOGRAM2D'].ax.tick_params(
            colors=text_color
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

        # Only update color_indices if changed
        target_color_indices = (
            selection_id_data if selection_id_data is not None else 0
        )

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

        if self._last_histogram_colormap != self.histogram_colormap:
            selected_histogram_colormap = colormaps.ALL_COLORMAPS[
                self.histogram_colormap
            ]
            selected_histogram_colormap = LinearSegmentedColormap.from_list(
                self.histogram_colormap,
                selected_histogram_colormap.colors,
            )
            histogram_artist.histogram_colormap = selected_histogram_colormap
            self._last_histogram_colormap = self.histogram_colormap
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

    def _update_colorbar(self, colormap):
        """Update or create colorbar for histogram plot."""
        self._remove_colorbar()

        if self.plot_type == 'HISTOGRAM2D':
            self.cax = self.canvas_widget.artists['HISTOGRAM2D'].ax.inset_axes(
                [1.05, 0, 0.05, 1]
            )
            self.colorbar = Colorbar(
                ax=self.cax,
                cmap=colormap,
                norm=self.canvas_widget.artists[
                    'HISTOGRAM2D'
                ]._get_normalization(
                    self.canvas_widget.artists['HISTOGRAM2D'].histogram[0],
                    is_overlay=False,
                ),
            )
            self.set_colorbar_style(color="white")

    def _remove_colorbar(self):
        """Remove colorbar if it exists."""
        if self.colorbar is not None:
            self.colorbar.remove()
            self.colorbar = None

    def _update_plot_elements(self):
        """Update common plot elements like semicircle, axes, etc."""
        if self.toggle_semi_circle:
            self._update_semi_circle_plot(self.canvas_widget.axes)

        self._enforce_axes_aspect()
        self._update_plot_bg_color()

    def plot(self, x_data=None, y_data=None, selection_id_data=None):
        """Plot the selected phasor features efficiently."""
        if not self.has_phasor_data():
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

            if hasattr(self, 'lifetime_tab') and selection_id_data is None:
                self.lifetime_tab.reapply_if_active()

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

        if plot_type == 'HISTOGRAM2D':
            self._update_histogram_plot(x_data, y_data, selection_id_data)
        elif plot_type == 'SCATTER':
            self._remove_colorbar()
            self._update_scatter_plot(x_data, y_data, selection_id_data)

        current_active = getattr(self.canvas_widget, 'active_artist', None)

        if (
            current_active != plot_type
            and plot_type in self.canvas_widget.artists
        ):
            self.canvas_widget.active_artist = plot_type
        elif plot_type not in self.canvas_widget.artists:
            return

        self._update_plot_elements()

    def set_colorbar_style(self, color="white"):
        """Set the colorbar style in the canvas widget."""
        self.colorbar.ax.yaxis.set_tick_params(color=color)
        self.colorbar.outline.set_edgecolor(color)
        if isinstance(self.colorbar.norm, LogNorm):
            self.colorbar.ax.set_ylabel("Log10(Count)", color=color)
        else:
            self.colorbar.ax.set_ylabel("Count", color=color)
        ticks = self.colorbar.ax.get_yticks()
        self.colorbar.ax.yaxis.set_major_locator(ticker.FixedLocator(ticks))
        tick_labels = self.colorbar.ax.get_yticklabels()
        for tick_label in tick_labels:
            tick_label.set_color(color)

    def closeEvent(self, event):
        """Clean up signal connections and child widgets before closing."""
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
            'lifetime_tab',
            'fret_tab',
        ):
            tab = getattr(self, tab_name, None)
            if tab is not None:
                with contextlib.suppress(Exception):
                    tab.close()

        super().closeEvent(event)
