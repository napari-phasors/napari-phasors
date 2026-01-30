import copy
import math
import warnings
from pathlib import Path

import matplotlib.ticker as ticker
import numpy as np
from biaplotter.plotter import CanvasWidget
from matplotlib.colorbar import Colorbar
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from napari.layers import Image, Labels, Shapes
from napari.utils import colormaps, notifications
from phasorpy.lifetime import phasor_from_lifetime
from qtpy import uic
from qtpy.QtCore import Qt, QTimer, Signal
from qtpy.QtGui import QStandardItem, QStandardItemModel
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QStyledItemDelegate,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ._utils import update_frequency_in_metadata
from .calibration_tab import CalibrationWidget
from .components_tab import ComponentsWidget
from .filter_tab import FilterWidget
from .fret_tab import FretWidget
from .lifetime_tab import LifetimeWidget
from .selection_tab import SelectionWidget


class CheckableComboBox(QComboBox):
    """A ComboBox with checkable items for multi-selection.

    Displays selected items as comma-separated text and emits
    selectionChanged signal when items are checked/unchecked.
    """

    selectionChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setModel(QStandardItemModel(self))
        self.setEditable(True)
        self.lineEdit().setReadOnly(True)
        self.lineEdit().setPlaceholderText("Select layers...")

        # Use a delegate to prevent closing on click
        self.setItemDelegate(QStyledItemDelegate(self))

        # Connect model signals
        self.model().dataChanged.connect(self._on_data_changed)

        # Track if we're inside the popup
        self._popup_visible = False

        # Make the line edit clickable to open popup
        self.lineEdit().installEventFilter(self)
        # Prevent cursor positioning in line edit
        self.lineEdit().setFocusPolicy(Qt.NoFocus)

    def eventFilter(self, obj, event):
        """Filter events to make line edit clickable."""
        if obj == self.lineEdit():
            if event.type() == event.MouseButtonRelease:
                # Toggle popup on mouse release
                if not self.view().isVisible():
                    self.showPopup()
                return True
            elif event.type() == event.MouseButtonPress:
                # Consume press event to prevent default behavior
                return True
        return super().eventFilter(obj, event)

    def addItem(self, text, checked=False):
        """Add a checkable item to the combobox."""
        item = QStandardItem(text)
        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
        item.setData(Qt.Checked if checked else Qt.Unchecked, Qt.CheckStateRole)
        self.model().appendRow(item)

    def addItems(self, texts):
        """Add multiple items to the combobox."""
        for text in texts:
            self.addItem(text)

    def clear(self):
        """Clear all items."""
        self.model().clear()
        self._update_display_text()

    def checkedItems(self):
        """Return list of checked item texts."""
        checked = []
        for i in range(self.model().rowCount()):
            item = self.model().item(i)
            if item.checkState() == Qt.Checked:
                checked.append(item.text())
        return checked

    def setCheckedItems(self, texts):
        """Set which items are checked by their text."""
        self.blockSignals(True)
        for i in range(self.model().rowCount()):
            item = self.model().item(i)
            if item.text() in texts:
                item.setCheckState(Qt.Checked)
            else:
                item.setCheckState(Qt.Unchecked)
        self.blockSignals(False)
        self._update_display_text()

    def _on_data_changed(self, topLeft, bottomRight, roles):
        """Handle item check state changes."""
        if Qt.CheckStateRole in roles:
            self._update_display_text()
            self.selectionChanged.emit()

    def _update_display_text(self):
        """Update the display text to show checked items."""
        checked = self.checkedItems()
        if not checked:
            self.lineEdit().setText("")
            self.lineEdit().setPlaceholderText("Select layers...")
        elif len(checked) == 1:
            self.lineEdit().setText(checked[0])
        else:
            self.lineEdit().setText(f"{len(checked)} layers selected")

    def showPopup(self):
        """Show the popup and track visibility."""
        self._popup_visible = True
        super().showPopup()

    def hidePopup(self):
        """Hide the popup."""
        self._popup_visible = False
        super().hidePopup()

    def itemCheckState(self, index):
        """Get the check state of item at index."""
        item = self.model().item(index)
        return item.checkState() if item else Qt.Unchecked

    def setItemCheckState(self, index, state):
        """Set the check state of item at index."""
        item = self.model().item(index)
        if item:
            item.setCheckState(state)


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

        self.setLayout(QVBoxLayout())

        # Initialize data attributes
        self._g_array = None
        self._s_array = None
        self._g_original_array = None
        self._s_original_array = None
        self._harmonics_array = None

        # Create top widget for canvas
        canvas_container = QWidget()
        canvas_container.setLayout(QVBoxLayout())
        self.layout().addWidget(canvas_container)

        # Load canvas widget (fixed at the top)
        self.canvas_widget = CanvasWidget(
            napari_viewer, highlight_enabled=False
        )
        self.canvas_widget.axes.set_aspect(1, adjustable='box')
        self.canvas_widget.setMinimumSize(300, 300)
        self.set_axes_labels()
        canvas_container.layout().addWidget(self.canvas_widget)

        # Create bottom widget for controls
        controls_container = QWidget()
        controls_container.setLayout(QVBoxLayout())
        self.layout().addWidget(controls_container)

        # Add checkable combobox for multi-layer selection
        image_layer_layout = QHBoxLayout()
        image_layer_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins
        image_layer_layout.setSpacing(5)  # Reduce spacing between widgets
        image_layer_layout.addWidget(QLabel("Image Layers:"))
        self.image_layers_checkable_combobox = CheckableComboBox()
        self.image_layers_checkable_combobox.setMaximumHeight(25)
        self.image_layers_checkable_combobox.setToolTip(
            "Select one or more layers to plot. Check multiple layers to merge their phasor data in the plot."
        )
        image_layer_layout.addWidget(self.image_layers_checkable_combobox, 1)

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
        self.harmonic_spinbox.setMaximumHeight(25)
        harmonics_and_mask_container.addWidget(self.harmonic_spinbox, 1)

        # Mask label and combobox
        self.mask_layer_label = QLabel("Mask Layer:")
        harmonics_and_mask_container.addWidget(self.mask_layer_label)
        self.mask_layer_combobox = QComboBox()
        self.mask_layer_combobox.setToolTip(
            "Create or select a Labels or Shapes layer with a mask to restrict analysis to specific regions. "
            "Selecting 'None' will disable masking."
        )
        self.mask_layer_combobox.addItem("None")
        self.mask_layer_combobox.setMaximumHeight(25)
        harmonics_and_mask_container.addWidget(self.mask_layer_combobox, 1)

        controls_container.layout().addLayout(harmonics_and_mask_container)

        # Add import buttons below harmonic spinbox
        import_buttons_layout = QHBoxLayout()
        import_buttons_layout.setContentsMargins(0, 0, 0, 0)
        import_buttons_layout.setSpacing(5)

        import_label = QLabel("Load and Apply Settings from:")
        import_buttons_layout.addWidget(import_label)

        self.import_from_layer_button = QPushButton("Layer")
        self.import_from_layer_button.setMaximumHeight(25)
        import_buttons_layout.addWidget(self.import_from_layer_button)

        self.import_from_file_button = QPushButton("OME-TIFF File")
        self.import_from_file_button.setMaximumHeight(25)
        import_buttons_layout.addWidget(self.import_from_file_button)

        import_buttons_widget = QWidget()
        import_buttons_widget.setLayout(import_buttons_layout)
        controls_container.layout().addWidget(import_buttons_widget)

        # Create tab widget
        self.tab_widget = QTabWidget()

        # Create a separate widget for the tabs to allow independent docking
        self.analysis_widget = QWidget()
        self.analysis_widget.setLayout(QVBoxLayout())
        self.analysis_widget.layout().addWidget(self.tab_widget)

        # Add the analysis widget to the viewer with a delay to ensure correct ordering
        QTimer.singleShot(100, self._add_analysis_dock_widget)

        canvas_container.setMinimumHeight(300)
        controls_container.setMinimumHeight(100)

        # Add a flag to prevent recursive calls
        self._updating_plot = False
        self._updating_settings = False

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
        self.image_layers_checkable_combobox.selectionChanged.connect(
            self.on_image_layer_changed
        )
        # Update all frequency widgets from layer metadata if layer changes
        self.image_layers_checkable_combobox.selectionChanged.connect(
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
        """Add the analysis widget to the viewer as a dock widget."""
        if (
            hasattr(self.viewer, 'window')
            and self.viewer.window is not None
            and hasattr(self.viewer.window, '_qt_window')
        ):
            self.viewer.window.add_dock_widget(
                self.analysis_widget, name="Phasor Analysis", area="right"
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
        """Get the name of the primary (first) selected layer.

        The primary layer is used for metadata operations and analysis.
        For backward compatibility, this returns the first selected layer
        or an empty string if no layer is selected.

        Returns
        -------
        str
            Name of the primary selected layer, or empty string if none.
        """
        selected = self.get_selected_layer_names()
        return selected[0] if selected else ""

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
                    if isinstance(mask_l, Labels) or isinstance(mask_l, Shapes)
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
                "Filter/Threshold",
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
            if source_settings is None:
                show_tab = True
            elif settings_key is None:
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
        """Restore only selected tab analyses."""
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
            self._apply_calibration_if_needed()
        if "filter_tab" in selected_tabs and hasattr(self, 'filter_tab'):
            self.filter_tab._on_image_layer_changed()
            self.filter_tab.apply_button_clicked()
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
                    description = json.loads(attrs["description"])
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
        except Exception as e:
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
        except Exception as e:
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

    def _on_tab_changed(self, index):
        """Handle tab change events to show/hide tab-specific lines."""
        current_tab = self.tab_widget.widget(index)

        self._hide_all_tab_artists()

        self._show_tab_artists(current_tab)

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

    def _on_semi_circle_changed(self, state):
        """Callback for semi circle checkbox change."""
        self._update_setting_in_metadata('semi_circle', bool(state))
        if not self._updating_settings:
            self.toggle_semi_circle = bool(state)

    def _on_harmonic_changed(self, value):
        """Callback for harmonic spinbox change."""
        self._update_setting_in_metadata('harmonic', value)
        if not self._updating_settings:
            self.refresh_current_plot()

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
        """Callback for bins change."""
        self._update_setting_in_metadata('number_of_bins', value)
        if not self._updating_settings:
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
        self.tab_widget.addTab(self.filter_tab, "Filter/Threshold")

    def _create_selection_tab(self):
        """Create the Cursor selection tab."""
        self.selection_tab = SelectionWidget(self.viewer, parent=self)
        self.tab_widget.addTab(self.selection_tab, "Selection")

    def _create_components_tab(self):
        """Create the Components tab."""
        self.components_tab = ComponentsWidget(self.viewer, parent=self)
        self.tab_widget.addTab(self.components_tab, "Components")

    def _create_lifetime_tab(self):
        """Create the Lifetime tab."""
        self.lifetime_tab = LifetimeWidget(self.viewer, parent=self)
        self.tab_widget.addTab(self.lifetime_tab, "Lifetime")

        self.harmonic_spinbox.valueChanged.connect(
            self.lifetime_tab._on_harmonic_changed
        )
        self.lifetime_tab.frequency_input.editingFinished.connect(
            lambda: self._broadcast_frequency_value_across_tabs(
                self.lifetime_tab.frequency_input.text()
            )
        )

    def _create_fret_tab(self):
        """Create the FRET tab."""
        self.fret_tab = FretWidget(self.viewer, parent=self)
        self.tab_widget.addTab(self.fret_tab, "FRET")
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
            except:
                continue

        for i, lifetime in enumerate(lifetimes):
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
            )[0]
            self.semi_circle_plot_artist_list.append(tick_line)

            if lifetime == 0:
                label_text = "0"
            else:
                label_text = f"{lifetime:g}"

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
            )
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
                return settings["frequency"]

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
            if self.white_background:
                color = "white"
            else:
                color = "none"  # Transparent background

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
    def plot_type(self, type):
        """Sets the plot type from the plot type combobox."""
        self.plotter_inputs_widget.plot_type_combobox.setCurrentText(type)

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
        if colormap not in colormaps.ALL_COLORMAPS.keys():
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
        try:
            self.canvas_widget.artists[
                'SCATTER'
            ].color_indices_changed_signal.disconnect(
                self.selection_tab.manual_selection_changed
            )
        except (TypeError, AttributeError):
            pass

        try:
            self.canvas_widget.artists[
                'HISTOGRAM2D'
            ].color_indices_changed_signal.disconnect(
                self.selection_tab.manual_selection_changed
            )
        except (TypeError, AttributeError):
            pass

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
                and "G" in layer.metadata.keys()
                and "S" in layer.metadata.keys()
                and "G_original" in layer.metadata.keys()
                and "S_original" in layer.metadata.keys()
            ]
            mask_layer_names = [
                layer.name
                for layer in self.viewer.layers
                if isinstance(layer, Labels) or isinstance(layer, Shapes)
            ]

            # Add items to the checkable combobox
            for name in layer_names:
                # Check if this layer was previously selected
                checked = name in previously_selected
                self.image_layers_checkable_combobox.addItem(name, checked)

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

            self.image_layers_checkable_combobox.blockSignals(False)
            self.mask_layer_combobox.blockSignals(False)

            # If mask layer was deleted, trigger the cleanup
            if mask_layer_was_deleted:
                self._on_mask_layer_changed("None")

            # Connect layer name change events (disconnect first to avoid duplicates)
            for layer_name in layer_names + mask_layer_names:
                layer = self.viewer.layers[layer_name]
                if (
                    isinstance(layer, Image)
                    and "phasor_features_labels_layer" in layer.metadata.keys()
                ):
                    try:
                        layer.events.name.disconnect(self.reset_layer_choices)
                    except (TypeError, ValueError):
                        pass  # Not connected, ignore
                    layer.events.name.connect(self.reset_layer_choices)
                if isinstance(layer, Shapes):
                    try:
                        layer.events.data.disconnect(
                            self._on_mask_data_changed
                        )
                    except (TypeError, ValueError):
                        pass  # Not connected, ignore
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
        """
        if getattr(self, "_in_on_image_layer_changed", False):
            return
        self._in_on_image_layer_changed = True
        try:
            selected_layers = self.get_selected_layers()

            # Update grid view based on selection
            self._update_grid_view(selected_layers)

            layer_name = self.get_primary_layer_name()
            if layer_name == "":
                self._g_array = None
                self._s_array = None
                self._harmonics_array = None
                return

            layer = self.viewer.layers[layer_name]
            layer_metadata = layer.metadata

            # Retrieve arrays from metadata (primary layer for backward compat)
            self._g_array = layer_metadata.get("G")
            self._s_array = layer_metadata.get("S")
            self._g_original_array = layer_metadata.get("G_original")
            self._s_original_array = layer_metadata.get("S_original")
            self._harmonics_array = layer_metadata.get("harmonics")

            # Compute intersection of harmonics across all selected layers
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

            # Reset mask layer combobox to "None" when image layer changes
            self.mask_layer_combobox.blockSignals(True)
            self.mask_layer_combobox.setCurrentText("None")
            self.mask_layer_combobox.blockSignals(False)

            self._initialize_plot_settings_in_metadata(layer)

            self._restore_plot_settings_from_metadata()

            self._sync_frequency_inputs_from_metadata()

            # Update filter widget when layer changes
            if hasattr(self, 'filter_tab'):
                self.filter_tab._on_image_layer_changed()

            # Update calibration button state when layer changes
            if hasattr(self, 'calibration_tab'):
                self.calibration_tab._on_image_layer_changed()

            # Update selection tab when layer changes
            if hasattr(self, 'selection_tab'):
                self.selection_tab._on_image_layer_changed()

            # Update lifetime tab when layer changes
            if hasattr(self, 'lifetime_tab'):
                self.lifetime_tab._on_image_layer_changed()

            # Update components tab when layer changes
            if hasattr(self, 'components_tab'):
                self.components_tab._on_image_layer_changed()

            # Update FRET tab when layer changes
            if hasattr(self, 'fret_tab'):
                self.fret_tab._on_image_layer_changed()

            self.plot()

        finally:
            self._in_on_image_layer_changed = False

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
            # Enable grid mode for multiple layers
            self.viewer.grid.enabled = True

            # Make selected layers visible, hide others (only phasor layers)
            for layer in self.viewer.layers:
                if isinstance(layer, Image) and "G" in layer.metadata:
                    layer.visible = layer.name in selected_names
        else:
            # Disable grid mode for single selection
            self.viewer.grid.enabled = False

            # Make the selected layer visible
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

        # Apply mask to image data (set values outside mask to NaN)
        mask_invalid = mask_data <= 0
        image_layer.data = np.where(mask_invalid, np.nan, image_layer.data)

        # Apply mask to G and S arrays
        # G and S have shape (n_harmonics, Y, X) or (Y, X)
        # mask_data has shape (Y, X)
        # We need to broadcast the mask across harmonics if needed

        g_array = image_layer.metadata['G']
        s_array = image_layer.metadata['S']

        if g_array.ndim == 3:
            # Multi-harmonic case: shape is (n_harmonics, Y, X)
            # Expand mask to (1, Y, X) for broadcasting
            mask_invalid_expanded = mask_invalid[np.newaxis, :, :]
            image_layer.metadata['G'] = np.where(
                mask_invalid_expanded, np.nan, g_array
            )
            image_layer.metadata['S'] = np.where(
                mask_invalid_expanded, np.nan, s_array
            )
        else:
            # Single harmonic case: shape is (Y, X)
            image_layer.metadata['G'] = np.where(mask_invalid, np.nan, g_array)
            image_layer.metadata['S'] = np.where(mask_invalid, np.nan, s_array)

    def _on_mask_layer_changed(self, text):
        """Handle changes to the mask layer combo box."""
        current_image_layer_name = (
            self.image_layer_with_phasor_features_combobox.currentText()
        )
        if not current_image_layer_name:
            return

        current_image_layer = self.viewer.layers[current_image_layer_name]

        # Restore original G and S and image data
        self._restore_original_phasor_data(current_image_layer)

        if text == "None":
            if 'mask' in current_image_layer.metadata:
                del current_image_layer.metadata['mask']
        else:
            mask_layer = self.viewer.layers[text]
            self._apply_mask_to_phasor_data(mask_layer, current_image_layer)

        # Update filter widget when layer changes
        if hasattr(self, 'filter_tab'):
            self.filter_tab._on_image_layer_changed()
            # Re-apply filter if previously applied
            if (
                current_image_layer.metadata['settings'].get('filter', None)
                is not None
                and current_image_layer.metadata['settings'].get(
                    'threshold', None
                )
                is not None
            ):
                self.filter_tab.apply_button_clicked()

        self.plot()

    def _on_mask_data_changed(self, event):
        """Handle changes to the mask layer data."""
        if self.mask_layer_combobox.currentText() != event.source.name:
            return
        current_image_layer_name = (
            self.image_layer_with_phasor_features_combobox.currentText()
        )
        current_image_layer = self.viewer.layers[current_image_layer_name]
        # Restore original G and S and image data (this also clears previously applied filters)
        self._restore_original_phasor_data(current_image_layer)
        mask_layer = event.source
        self._apply_mask_to_phasor_data(mask_layer, current_image_layer)

        # Apply changes to the filter tab
        if hasattr(self, 'filter_tab'):
            self.filter_tab._on_image_layer_changed()
            # Re-apply filter if previously applied
            if (
                current_image_layer.metadata['settings'].get('filter', None)
                is not None
                and current_image_layer.metadata['settings'].get(
                    'threshold', None
                )
                is not None
            ):
                self.filter_tab.apply_button_clicked()

        self.refresh_current_plot()

    def refresh_phasor_data(self):
        """Reload phasor data from the current layer metadata and replot."""
        layer_name = (
            self.image_layer_with_phasor_features_combobox.currentText()
        )
        if not layer_name or layer_name not in self.viewer.layers:
            return

        layer = self.viewer.layers[layer_name]
        layer_metadata = layer.metadata

        # Retrieve arrays from metadata
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
        # Check primary layer arrays (backward compatibility)
        if (
            self._g_array is not None
            and self._s_array is not None
            and self._harmonics_array is not None
        ):
            return True

        # Also check if any selected layer has phasor data
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
        except Exception:
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

            # Get harmonic index for this layer
            if harmonics_array is not None:
                harmonics_array = np.atleast_1d(harmonics_array)
                target_harmonic = self.harmonic
                try:
                    harmonic_idx = int(
                        np.where(harmonics_array == target_harmonic)[0][0]
                    )
                except (IndexError, ValueError):
                    # Harmonic not found in this layer, skip it
                    continue
            else:
                harmonic_idx = 0

            # Extract data for the harmonic
            if g_array.ndim == 3:
                g = g_array[harmonic_idx]
                s = s_array[harmonic_idx]
            else:
                g = g_array
                s = s_array

            # Flatten and filter valid values
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

    def set_axes_labels(self):
        """Set the axes labels in the canvas widget."""
        text_color = "white"

        self.canvas_widget.artists['SCATTER'].ax.set_xlabel(
            "G", color=text_color
        )
        self.canvas_widget.artists['SCATTER'].ax.set_ylabel(
            "S", color=text_color
        )
        self.canvas_widget.artists['HISTOGRAM2D'].ax.set_xlabel(
            "G", color=text_color
        )
        self.canvas_widget.artists['HISTOGRAM2D'].ax.set_ylabel(
            "S", color=text_color
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

        if selection_id_data is not None:
            self.canvas_widget.artists['SCATTER'].color_indices = (
                selection_id_data
            )
        else:
            self.canvas_widget.artists['SCATTER'].color_indices = 0

    def _update_histogram_plot(self, x_data, y_data, selection_id_data=None):
        """Update the histogram plot with new data."""
        plot_data = np.column_stack((x_data, y_data))

        # Configure histogram artist properties
        self.canvas_widget.artists['HISTOGRAM2D'].data = plot_data
        self.canvas_widget.artists['HISTOGRAM2D'].cmin = 1
        self.canvas_widget.artists['HISTOGRAM2D'].bins = self.histogram_bins

        # Set colormap
        selected_histogram_colormap = colormaps.ALL_COLORMAPS[
            self.histogram_colormap
        ]
        selected_histogram_colormap = LinearSegmentedColormap.from_list(
            self.histogram_colormap,
            selected_histogram_colormap.colors,
        )
        self.canvas_widget.artists['HISTOGRAM2D'].histogram_colormap = (
            selected_histogram_colormap
        )

        # Set normalization method
        if self.canvas_widget.artists['HISTOGRAM2D'].histogram is not None:
            if self.histogram_log_scale:
                self.canvas_widget.artists[
                    'HISTOGRAM2D'
                ].histogram_color_normalization_method = "log"
            else:
                self.canvas_widget.artists[
                    'HISTOGRAM2D'
                ].histogram_color_normalization_method = "linear"

        # Apply selection data if available
        if selection_id_data is not None:
            self.canvas_widget.artists['HISTOGRAM2D'].color_indices = (
                selection_id_data
            )
        else:
            self.canvas_widget.artists['HISTOGRAM2D'].color_indices = 0

        # Update colorbar for histogram
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

            # Restore selection visualization if active
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
