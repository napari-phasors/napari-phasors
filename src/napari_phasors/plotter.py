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
from qtpy.QtCore import Qt
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
    QSplitter,
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
    image_layer_with_phasor_features_combobox : QComboBox
        The combobox for selecting the image layer with phasor features.
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
        self._labels_layer_with_phasor_features = None

        # Create a splitter to separate canvas from controls
        splitter = QSplitter(Qt.Vertical)
        self.layout().addWidget(splitter)

        # Create top widget for canvas
        canvas_container = QWidget()
        canvas_container.setLayout(QVBoxLayout())

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

        # Add select image combobox
        image_layer_layout = QHBoxLayout()
        image_layer_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins
        image_layer_layout.setSpacing(5)  # Reduce spacing between widgets
        image_layer_layout.addWidget(QLabel("Image Layer:"))
        self.image_layer_with_phasor_features_combobox = QComboBox()
        self.image_layer_with_phasor_features_combobox.setMaximumHeight(
            25
        )  # Set smaller height
        image_layer_layout.addWidget(
            self.image_layer_with_phasor_features_combobox, 1
        )  # Add stretch factor of 1

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
        self.harmonic_spinbox.setMaximumHeight(25)  # Set smaller height
        harmonics_and_mask_container.addWidget(
            self.harmonic_spinbox, 1
        )  # Add stretch factor of 1

        # Mask label and combobox (right side)
        self.mask_layer_label = QLabel("Mask Layer:")
        harmonics_and_mask_container.addWidget(self.mask_layer_label)
        self.mask_layer_combobox = QComboBox()
        self.mask_layer_combobox.setToolTip(
            "Create or select a Labels or Shapes layer with a mask to restrict analysis to specific regions. "
            "Selecting 'None' will disable masking."
        )
        self.mask_layer_combobox.addItem("None")
        self.mask_layer_combobox.setMaximumHeight(25)  # Set smaller height
        harmonics_and_mask_container.addWidget(self.mask_layer_combobox, 1)

        controls_container.layout().addLayout(harmonics_and_mask_container)
        self.mask = None

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

        # Create tab widget
        self.tab_widget = QTabWidget()
        controls_container.layout().addWidget(self.tab_widget)

        # Add widgets to splitter
        splitter.addWidget(canvas_container)
        splitter.addWidget(controls_container)

        # Configure splitter to prevent overlap
        splitter.setStretchFactor(0, 1)  # Canvas gets priority
        splitter.setStretchFactor(1, 0)  # Controls maintain minimum size
        splitter.setCollapsible(0, False)  # Canvas cannot be collapsed
        splitter.setCollapsible(1, True)  # Controls can be collapsed if needed
        splitter.setStyleSheet(
            """
            QSplitter::handle {
                background: #414851;         /* default state */
            }
            QSplitter::handle:hover {
                background: #414851;         /* on hover */
            }
            QSplitter::handle:pressed {
                background: #7b8d8e;         /* while dragging */
            }
        """
        )

        canvas_container.setMinimumHeight(300)
        controls_container.setMinimumHeight(300)
        splitter.setSizes([800, 400])

        # Add a flag to prevent recursive calls
        self._updating_plot = False
        self._updating_settings = False

        # Create Settings tab
        self.settings_tab = QWidget()
        self.settings_tab.setLayout(QVBoxLayout())
        self.tab_widget.addTab(self.settings_tab, "Plot Settings")

        # Load plotter inputs widget from ui file (moved to Settings tab)
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
        self.image_layer_with_phasor_features_combobox.currentIndexChanged.connect(
            self.on_labels_layer_with_phasor_features_changed
        )
        # Update all frequency widgets from layer metadata if layer changes
        self.image_layer_with_phasor_features_combobox.currentIndexChanged.connect(
            self._sync_frequency_inputs_from_metadata
        )
        # Update mask when mask layer selection changes
        self.mask_layer_combobox.currentTextChanged.connect(
            self.on_mask_layer_changed
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
        self.histogram_colormap = (
            "turbo"  # Set default colormap (same as in biaplotter)
        )

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

    def _get_default_plot_settings(self):
        """Get default settings dictionary for plot parameters."""

        default_harmonic = 1  # fallback default
        layer_name = (
            self.image_layer_with_phasor_features_combobox.currentText()
        )
        if layer_name:
            try:
                layer_metadata = self.viewer.layers[layer_name].metadata
                if "phasor_features_labels_layer" in layer_metadata:
                    phasor_features = layer_metadata[
                        "phasor_features_labels_layer"
                    ]
                    if (
                        phasor_features.features is not None
                        and "harmonic" in phasor_features.features.columns
                    ):
                        default_harmonic = int(
                            phasor_features.features["harmonic"].min()
                        )
            except (KeyError, AttributeError, ValueError):
                pass

        return {
            'harmonic': default_harmonic,
            'semi_circle': True,
            'white_background': True,
            'plot_type': 'HISTOGRAM2D',
            'colormap': 'turbo',
            'number_of_bins': 150,
            'log_scale': False,
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

        layer = self.viewer.layers[layer_name]
        if 'settings' not in layer.metadata:
            self._initialize_plot_settings_in_metadata(layer)
            return

        self._updating_settings = True
        try:
            settings = layer.metadata['settings']

            # Restore harmonic
            if 'harmonic' in settings:
                self.harmonic_spinbox.setValue(settings['harmonic'])

            # Restore white background first (this affects circle colors)
            if 'white_background' in settings:
                self.plotter_inputs_widget.white_background_checkbox.setChecked(
                    settings['white_background']
                )
                # Update axes labels and plot background immediately
                self.set_axes_labels()
                self._update_plot_bg_color()

            # Restore semi circle (this will update the circle/semicircle with correct colors)
            if 'semi_circle' in settings:
                # Use the setter to properly update the display
                self.toggle_semi_circle = settings['semi_circle']

            # Restore plot type
            if 'plot_type' in settings:
                self.plotter_inputs_widget.plot_type_combobox.setCurrentText(
                    settings['plot_type']
                )

            # Restore colormap
            if 'colormap' in settings:
                self.plotter_inputs_widget.colormap_combobox.setCurrentText(
                    settings['colormap']
                )

            # Restore number of bins
            if 'number_of_bins' in settings:
                self.plotter_inputs_widget.number_of_bins_spinbox.setValue(
                    settings['number_of_bins']
                )

            # Restore log scale
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
            ("Plot Settings", "settings_tab", None),  # Always has settings
            ("Calibration", "calibration_tab", "calibrated"),
            (
                "Filter/Threshold",
                "filter_tab",
                ["threshold", "filter"],
            ),  # Either threshold or filter
            ("Lifetime", "lifetime_tab", "lifetime"),
            ("FRET", "fret_tab", "fret"),
            ("Components", "components_tab", "component_analysis"),
        ]

        checkboxes = {}

        for label, attr, settings_key in tab_mapping:
            # Determine if this tab should be shown
            show_tab = False

            if source_settings is None:
                # No settings provided, show all tabs
                show_tab = True
            elif settings_key is None:
                # Always show (e.g., Plot Settings)
                show_tab = True
            elif isinstance(settings_key, list):
                # Show if any of the keys exist (e.g., Filter/Threshold)
                show_tab = any(key in source_settings for key in settings_key)
            else:
                # Show if the specific key exists
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
            and "phasor_features_labels_layer" in layer.metadata
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

            source_settings = source_layer.metadata.get('settings', {})

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
        import copy

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
        if hasattr(self, 'components_tab'):
            self._set_components_visibility(False)

        if hasattr(self, 'fret_tab'):
            self._set_fret_visibility(False)

    def _show_tab_artists(self, current_tab):
        """Show artists for the specified tab."""
        if current_tab == getattr(self, 'components_tab', None):
            self._set_components_visibility(True)

        elif current_tab == getattr(self, 'fret_tab', None):
            self._set_fret_visibility(True)

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
        """Reset the image layer with phasor features combobox choices."""
        if getattr(self, '_resetting_layer_choices', False):
            return

        self._resetting_layer_choices = True

        try:
            # Store current selection
            image_layer_combobox_current_text = (
                self.image_layer_with_phasor_features_combobox.currentText()
            )
            mask_layer_combobox_current_text = (
                self.mask_layer_combobox.currentText()
            )

            self.image_layer_with_phasor_features_combobox.blockSignals(True)
            self.mask_layer_combobox.blockSignals(True)

            self.image_layer_with_phasor_features_combobox.clear()
            self.mask_layer_combobox.clear()

            layer_names = [
                layer.name
                for layer in self.viewer.layers
                if isinstance(layer, Image)
                and "phasor_features_labels_layer" in layer.metadata.keys()
            ]
            mask_layer_names = [
                layer.name
                for layer in self.viewer.layers
                if isinstance(layer, Labels) or isinstance(layer, Shapes)
            ]

            self.image_layer_with_phasor_features_combobox.addItems(
                layer_names
            )
            self.mask_layer_combobox.addItems(["None"] + mask_layer_names)

            # Restore combobox selection if it still exists
            if image_layer_combobox_current_text in layer_names:
                self.image_layer_with_phasor_features_combobox.setCurrentText(
                    image_layer_combobox_current_text
                )
            if mask_layer_combobox_current_text in mask_layer_names:
                self.mask_layer_combobox.setCurrentText(
                    mask_layer_combobox_current_text
                )

            self.image_layer_with_phasor_features_combobox.blockSignals(False)
            self.mask_layer_combobox.blockSignals(False)

            # Connect layer name change events (disconnect first to avoid duplicates)
            for layer_name in layer_names + mask_layer_names:
                layer = self.viewer.layers[layer_name]
                if isinstance(layer, Image) and "phasor_features_labels_layer" in layer.metadata.keys():
                    try:
                        layer.events.name.disconnect(self.reset_layer_choices)
                    except (TypeError, ValueError):
                        pass # Not connected, ignore
                    layer.events.name.connect(self.reset_layer_choices)
                if isinstance(layer, Shapes):
                    try:
                        layer.events.data.disconnect(self.on_mask_data_changed)
                    except (TypeError, ValueError):
                        pass # Not connected, ignore
                    layer.events.data.connect(self.on_mask_data_changed)
                if isinstance(layer, Labels):
                    try:
                        layer.events.paint.disconnect(self.on_mask_data_changed)
                        layer.events.set_data.disconnect(self.on_mask_data_changed)
                    except (TypeError, ValueError):
                        pass # Not connected, ignore
                    layer.events.paint.connect(self.on_mask_data_changed)
                    layer.events.set_data.connect(self.on_mask_data_changed)

            new_text = (
                self.image_layer_with_phasor_features_combobox.currentText()
            )
            if new_text != image_layer_combobox_current_text  or (
                image_layer_combobox_current_text == "" and new_text != ""
            ):
                self.on_labels_layer_with_phasor_features_changed()
                self._sync_frequency_inputs_from_metadata()

        finally:
            self._resetting_layer_choices = False

    def on_labels_layer_with_phasor_features_changed(self):
        """Handle changes to the labels layer with phasor features."""
        if getattr(
            self, "_in_on_labels_layer_with_phasor_features_changed", False
        ):
            return
        self._in_on_labels_layer_with_phasor_features_changed = True
        try:
            labels_layer_name = (
                self.image_layer_with_phasor_features_combobox.currentText()
            )
            if labels_layer_name == "":
                self._labels_layer_with_phasor_features = None
                return
            layer_metadata = self.viewer.layers[labels_layer_name].metadata
            self._labels_layer_with_phasor_features = layer_metadata[
                "phasor_features_labels_layer"
            ]

            available_harmonics = (
                self._labels_layer_with_phasor_features.features[
                    "harmonic"
                ].unique()
            )
            min_harmonic = int(available_harmonics.min())
            max_harmonic = int(available_harmonics.max())

            self.harmonic_spinbox.setMinimum(min_harmonic)
            self.harmonic_spinbox.setMaximum(max_harmonic)

            layer = self.viewer.layers[labels_layer_name]
            self._initialize_plot_settings_in_metadata(layer)

            self._restore_plot_settings_from_metadata()

            self._sync_frequency_inputs_from_metadata()

            # Update filter widget when layer changes
            if hasattr(self, 'filter_tab'):
                self.filter_tab._on_image_layer_changed()

            # Update calibration button state when layer changes
            if hasattr(self, 'calibration_tab'):
                self.calibration_tab._on_image_layer_changed()

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
            self._in_on_labels_layer_with_phasor_features_changed = False

    def on_mask_layer_changed(self, text):
        """Handle changes to the mask layer combo box."""
        print(f"Mask layer changed to: {text}")
        # if text == "None":
        #     self.update_mask_column_and_plot(None)
        # else:
        #     mask_layer = self.viewer.layers[text]
        #     self.update_mask_column_and_plot(mask_layer)

    def on_mask_data_changed(self, event):
        """Handle changes to the mask layer data."""
        print(f"Mask layer data changed: {event}")
        # if self.mask_layer_combobox.currentText() != event.source.name:
        #     return
        # mask_layer = event.source
        # self.update_mask_column_and_plot(mask_layer)

    def get_features(self):
        """Get the G and S features for the selected harmonic and selection id.

        Returns
        -------
        x_data : np.ndarray
            The G feature data.
        y_data : np.ndarray
            The S feature data.
        selection_data : np.ndarray
            The selection data.
        """
        if self._labels_layer_with_phasor_features is None:
            return None
        if self._labels_layer_with_phasor_features.features is None:
            return None

        table = self._labels_layer_with_phasor_features.features
        x_data = table['G'][table['harmonic'] == self.harmonic].values
        y_data = table['S'][table['harmonic'] == self.harmonic].values
        mask = np.isnan(x_data) & np.isnan(y_data)
        x_data = x_data[~mask]
        y_data = y_data[~mask]

        if (
            self.selection_tab.selection_id is None
            or self.selection_tab.selection_id == ""
            or self.selection_tab.selection_id not in table.columns
        ):
            return x_data, y_data, None
        else:
            selection_data = table[self.selection_tab.selection_id][
                table['harmonic'] == self.harmonic
            ].values
            selection_data = selection_data[~mask]
            return x_data, y_data, selection_data

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
        if self._labels_layer_with_phasor_features is None:
            return

        if getattr(self, '_updating_plot', False):
            return

        self._updating_plot = True

        if x_data is None or y_data is None:
            features = self.get_features()
            if features is None:
                return
            x_data, y_data, selection_id_data = features

        if len(x_data) == 0 or len(y_data) == 0:
            return

        self._set_active_artist_and_plot(
            self.plot_type, x_data, y_data, selection_id_data
        )

        self._updating_plot = False

    def refresh_current_plot(self):
        """Refresh the current plot with existing data."""
        self.plot()

    def switch_plot_type(self, new_plot_type):
        """Switch between plot types efficiently."""
        self._connect_active_artist_signals()

        features = self.get_features()
        if features is not None:
            x_data, y_data, selection_id_data = features
            self._set_active_artist_and_plot(
                new_plot_type, x_data, y_data, selection_id_data
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
