import contextlib
import copy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from napari.layers import Image
from napari.utils.notifications import show_error, show_warning
from phasorpy.lifetime import (
    phasor_from_fret_donor,
    phasor_to_apparent_lifetime,
    phasor_to_normal_lifetime,
)
from phasorpy.phasor import phasor_center, phasor_nearest_neighbor
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor, QDoubleValidator
from qtpy.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from superqt import QToggleSwitch

from ._mapping_filters import (
    FRET_EFFICIENCY,
    MappingFilterList,
    baseline_arrays,
    combined_mask,
    get_filters,
    kept_fraction,
    normalize_filters,
    rebuild_layer_from_filters,
    set_filters,
)
from ._parallel import parallel_map
from ._settings_store import merge_keyed_path
from ._timelapse import slice_datasets
from ._utils import (
    AutoUpdateMixin,
    CheckableComboBox,
    CurrentPageStackedWidget,
    HistogramWidget,
    analysis_section_stylesheet,
    create_settings_note_label,
    layer_colormap_from_settings,
    make_section,
    set_settings_note,
    setup_primary_button,
)

_FRET_OUTPUT_METADATA_KEY = 'phasor_fret_output'


class FretWidget(AutoUpdateMixin, QWidget):
    """Widget to perform FLIM FRET analysis."""

    def __init__(self, viewer, parent=None):
        """Initialize the FretWidget."""
        super().__init__()
        self.viewer = viewer
        self.parent_widget = parent
        self.frequency = 80.0
        self.donor_lifetime = 2.0
        self._fret_efficiencies = np.linspace(0, 1, 500)
        self.current_donor_line = None
        self.fret_layer = (
            None  # Reference to first layer for backward compatibility
        )
        self.fret_layers = []  # List of all FRET efficiency layers
        # Set while this tab rewrites the phasor arrays from the filter
        # stack, so the refresh it triggers does not recurse back into it.
        self._applying_mapping_filter = False
        self.colormap_contrast_limits = None
        self.fret_colormap = None
        self.colormap_gamma = 1.0
        self._fret_range_initialized = False
        self._updating_linked_layers = (
            False  # Flag to prevent recursive updates
        )
        self.use_colormap = True
        self.colormap_density_factor = (
            5  # Controls trajectory colormap detail level
        )
        self.current_donor_circle = None
        self.current_background_circle = None
        self._updating_settings = False
        self._needs_update = False  # Deferred update flag
        self._name_event_layers = {}

        # Initialize parameters
        self.donor_background = 0.1
        self.background_real = 0.0
        self.background_imag = 0.0
        self.donor_fretting_proportion = 1.0

        # Store background positions for different harmonics
        self.background_positions_by_harmonic = {}

        # Track current harmonic
        self.current_harmonic = 1

        # Setup UI
        self.setup_ui()

        # Connect to layer events to update background combobox
        self.viewer.layers.events.inserted.connect(self._on_layer_changed)
        self.viewer.layers.events.removed.connect(self._on_layer_changed)

    def setup_ui(self):
        """Set up the user interface for the FRET widget with a scroll area."""
        # Create the scroll area and the content widget
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)
        self.setStyleSheet(analysis_section_stylesheet())

        def _make_form():
            """A compact, growing form layout used inside each section box."""
            f = QFormLayout()
            f.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
            return f

        # Donor section ------------------------------------------------------
        donor_box, donor_box_layout = make_section("Donor")
        form = _make_form()

        # Frequency
        freq_row = QHBoxLayout()
        self.frequency_input = QLineEdit()
        self.frequency_input.setPlaceholderText("Frequency (MHz)")
        self.frequency_input.setValidator(QDoubleValidator())
        self.frequency_input.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Fixed
        )
        self.frequency_input.textChanged.connect(self._on_parameters_changed)
        self.frequency_input.setToolTip(
            "Enter the laser pulse or modulation frequency in MHz"
        )
        freq_row.addWidget(self.frequency_input)
        form.addRow("Frequency (MHz):", freq_row)

        # Donor lifetime source selector
        donor_source_row = QHBoxLayout()
        self.donor_source_selector = QComboBox()
        self.donor_source_selector.addItems(["Manual", "From layer(s)"])
        self.donor_source_selector.setMinimumContentsLength(8)
        self.donor_source_selector.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon
        )
        self.donor_source_selector.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Fixed
        )
        self.donor_source_selector.currentIndexChanged.connect(
            self._on_donor_source_changed
        )
        self.donor_source_selector.setToolTip(
            "Select whether to input donor lifetime manually or derive it from a layer"
        )
        donor_source_row.addWidget(self.donor_source_selector)
        form.addRow("Donor lifetime source:", donor_source_row)

        # Donor lifetime stacked input
        self.donor_stack = CurrentPageStackedWidget()

        # Page 0: Manual lifetime
        donor_manual_page = QWidget()
        donor_manual_layout = QHBoxLayout(donor_manual_page)
        donor_manual_layout.setContentsMargins(0, 0, 0, 0)
        self.donor_line_edit = QLineEdit()
        self.donor_line_edit.setPlaceholderText("Donor Lifetime (ns)")
        self.donor_line_edit.setValidator(QDoubleValidator())
        self.donor_line_edit.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Fixed
        )
        self.donor_line_edit.textChanged.connect(self._on_parameters_changed)
        self.donor_line_edit.setToolTip(
            "Enter the donor lifetime in nanoseconds"
        )
        donor_manual_layout.addWidget(self.donor_line_edit)
        self.donor_stack.addWidget(donor_manual_page)

        # Page 1: From layer (combobox + mode)
        donor_layer_page = QWidget()
        donor_layer_layout = QHBoxLayout(donor_layer_page)
        donor_layer_layout.setContentsMargins(0, 0, 0, 0)

        self.donor_lifetime_combobox = CheckableComboBox(
            enable_primary_layer=False,
            placeholder="Select layer(s)...",
        )
        self.donor_lifetime_combobox.selectionChanged.connect(
            self._calculate_donor_lifetime
        )
        self.donor_lifetime_combobox.setToolTip(
            "Select one or more layers from which to derive the donor lifetime (averaged)"
        )
        self.lifetime_type_combobox = QComboBox()
        self.lifetime_type_combobox.setMinimumContentsLength(8)
        self.lifetime_type_combobox.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon
        )
        self.lifetime_type_combobox.addItems(
            [
                "Apparent Phase Lifetime",
                "Apparent Modulation Lifetime",
                "Normal Lifetime",
            ]
        )
        self.lifetime_type_combobox.currentIndexChanged.connect(
            self._calculate_donor_lifetime
        )
        self.lifetime_type_combobox.setToolTip(
            "Select the method to calculate donor lifetime from phasor coordinates"
        )
        donor_layer_layout.addWidget(self.donor_lifetime_combobox)
        donor_layer_layout.addWidget(self.lifetime_type_combobox)
        self.donor_stack.addWidget(donor_layer_page)

        # Dynamic donor label that changes based on mode
        self.donor_label = QLabel("Donor lifetime (ns):")
        form.addRow(self.donor_label, self.donor_stack)

        # Donor Background slider
        background_slider_layout = QHBoxLayout()
        self.background_slider = QSlider(Qt.Horizontal)
        self.background_slider.setMinimum(0)
        self.background_slider.setMaximum(100)
        self.background_slider.setValue(10)  # 0.1 default
        self.background_slider.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Fixed
        )
        self.background_slider.valueChanged.connect(
            self._on_background_slider_changed
        )
        self.background_slider.setToolTip(
            "Weight of background fluorescence in donor channel relative to fluorescence of donor without FRET. A weight of 1 means the fluorescence of background and donor without FRET are equal."
        )
        background_slider_layout.addWidget(self.background_slider)
        self.background_label = QLabel("0.10")
        self.background_label.setAlignment(Qt.AlignCenter)
        background_slider_layout.addWidget(self.background_label)
        form.addRow("Donor Background:", background_slider_layout)

        donor_box_layout.addLayout(form)
        layout.addWidget(donor_box)

        # Background section -------------------------------------------------
        bg_box, bg_box_layout = make_section("Background")
        form = _make_form()

        # Background position source selector
        bg_source_row = QHBoxLayout()
        self.bg_source_selector = QComboBox()
        self.bg_source_selector.addItems(["Manual", "From layer(s)"])
        self.bg_source_selector.setMinimumContentsLength(8)
        self.bg_source_selector.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon
        )
        self.bg_source_selector.currentIndexChanged.connect(
            self._on_bg_source_changed
        )
        self.bg_source_selector.setToolTip(
            "Select whether to input background position manually or derive it from a layer"
        )
        bg_source_row.addWidget(self.bg_source_selector)
        form.addRow("Background source:", bg_source_row)

        # Background stacked input
        self.bg_stack = CurrentPageStackedWidget()

        # Page 0: Manual G,S
        bg_manual_page = QWidget()
        bg_manual_layout = QHBoxLayout(bg_manual_page)
        bg_manual_layout.setContentsMargins(0, 0, 0, 0)
        bg_manual_layout.addWidget(QLabel("G:"))
        self.background_real_edit = QLineEdit()
        self.background_real_edit.setPlaceholderText("Real coordinate")
        self.background_real_edit.setValidator(QDoubleValidator())
        self.background_real_edit.setText("0.0")
        self.background_real_edit.textChanged.connect(
            self._on_background_position_changed
        )
        self.background_real_edit.setToolTip(
            "Real component of background fluorescence phasor coordinate at frequency"
        )
        bg_manual_layout.addWidget(self.background_real_edit)
        bg_manual_layout.addWidget(QLabel("S:"))
        self.background_imag_edit = QLineEdit()
        self.background_imag_edit.setPlaceholderText("Imaginary coordinate")
        self.background_imag_edit.setValidator(QDoubleValidator())
        self.background_imag_edit.setText("0.0")
        self.background_imag_edit.textChanged.connect(
            self._on_background_position_changed
        )
        self.background_imag_edit.setToolTip(
            "Imaginary component of background fluorescence phasor coordinate at frequency"
        )
        bg_manual_layout.addWidget(self.background_imag_edit)
        self.bg_stack.addWidget(bg_manual_page)

        # Page 1: From layer
        bg_image_page = QWidget()
        bg_image_layout = QHBoxLayout(bg_image_page)
        bg_image_layout.setContentsMargins(0, 0, 0, 0)

        self.background_image_combobox = CheckableComboBox(
            enable_primary_layer=False,
            placeholder="Select layer(s)...",
        )
        self.background_image_combobox.selectionChanged.connect(
            self._calculate_background_position
        )
        self.background_image_combobox.setToolTip(
            "Select one or more layers from which to derive the background position (averaged)"
        )
        bg_image_layout.addWidget(self.background_image_combobox)
        self.bg_stack.addWidget(bg_image_page)

        # Dynamic background label that changes based on mode
        self.background_position_label = QLabel("Background position:")
        form.addRow(self.background_position_label, self.bg_stack)

        bg_box_layout.addLayout(form)
        layout.addWidget(bg_box)

        # Fretting section ---------------------------------------------------
        fretting_box, fretting_box_layout = make_section("Fretting")
        form = _make_form()

        # Proportion of Donors Fretting slider and label
        fretting_layout = QHBoxLayout()
        self.fretting_slider = QSlider(Qt.Horizontal)
        self.fretting_slider.setMinimum(0)
        self.fretting_slider.setMaximum(100)
        self.fretting_slider.setValue(100)  # 1.0 default
        self.fretting_slider.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Fixed
        )
        self.fretting_slider.valueChanged.connect(
            self._on_fretting_slider_changed
        )
        self.fretting_slider.setToolTip(
            "Fraction of donors participating in FRET"
        )
        fretting_layout.addWidget(self.fretting_slider)
        self.fretting_label = QLabel("1.00")
        self.fretting_label.setAlignment(Qt.AlignCenter)
        fretting_layout.addWidget(self.fretting_label)
        form.addRow("Proportion fretting:", fretting_layout)

        fretting_box_layout.addLayout(form)
        layout.addWidget(fretting_box)

        # Colormap over trajectory toggle
        self.colormap_checkbox = QToggleSwitch(
            "Overlay colormap on donor trajectory"
        )
        self.colormap_checkbox.onColor = QColor("#27ae60")  # Nice Green
        self.colormap_checkbox.setChecked(True)
        self.colormap_checkbox.toggled.connect(
            self._on_colormap_checkbox_changed
        )

        # Plot button
        self.calculate_fret_efficiency_button = QPushButton(
            "Calculate FRET efficiency"
        )
        self._refresh_calculate_button = setup_primary_button(
            self.calculate_fret_efficiency_button,
            self._fret_validation,
            self.calculate_fret_efficiency,
            ready_tooltip="Calculate FRET efficiency for the selected "
            "layer(s).",
        )
        # Cautions about settings and frequencies a Calculate would change.
        self._settings_note = create_settings_note_label(content_widget)
        layout.addWidget(self._settings_note)
        layout.addWidget(self.calculate_fret_efficiency_button)

        layout.addWidget(
            self._build_autoupdate_toggle(
                self.calculate_fret_efficiency_button,
                self._fret_validation,
                self.calculate_fret_efficiency,
                "Recalculate the FRET efficiency automatically whenever the "
                "donor lifetime, frequency, background, fretting proportion, "
                "layer selection, or filtered/calibrated phasor data change.",
            )
        )
        layout.addWidget(self.colormap_checkbox)

        # Filter section -----------------------------------------------------
        # One efficiency criterion, always shown and switched on and off with
        # its check box. It lives in the same per-layer stack the Phasor
        # Mapping tab edits, so the two compose; that tab's criteria are left
        # out of this list but are kept untouched in the stack.
        filter_box, filter_box_layout = make_section("Filter")
        self.filter_box = filter_box

        self.filter_intro_label = QLabel(
            "Discard pixels whose FRET efficiency falls outside a range."
        )
        self.filter_intro_label.setWordWrap(True)
        self.filter_intro_label.setToolTip(
            "The efficiency a filter tests is recomputed from the donor "
            "trajectory, and re-captured every time you recalculate, so the "
            "range always means the same thing as the map beside it."
        )
        filter_box_layout.addWidget(self.filter_intro_label)

        self.filter_list = MappingFilterList([FRET_EFFICIENCY], single=True)
        self.filter_list.set_params_provider(self._fret_filter_params)
        self.filter_list.set_harmonic_provider(self._current_harmonic)
        self.filter_list.filtersChanged.connect(self._on_filters_changed)
        filter_box_layout.addWidget(self.filter_list)
        layout.addWidget(filter_box)

        # Re-evaluate the button whenever a required input changes.
        self.frequency_input.textChanged.connect(
            lambda _=None: self._refresh_calculate_button()
        )
        self.donor_line_edit.textChanged.connect(
            lambda _=None: self._refresh_calculate_button()
        )
        self.frequency_input.textChanged.connect(
            lambda _=None: self._refresh_filter_enable_state()
        )
        self.donor_line_edit.textChanged.connect(
            lambda _=None: self._refresh_filter_enable_state()
        )
        # Autoupdate follows *committed* values -- a released slider, or a
        # text field the user left -- so a drag or a half-typed number does
        # not trigger one full recalculation per intermediate value.
        self.frequency_input.editingFinished.connect(self.request_autoupdate)
        self.donor_line_edit.editingFinished.connect(self.request_autoupdate)
        self.background_real_edit.editingFinished.connect(
            self.request_autoupdate
        )
        self.background_imag_edit.editingFinished.connect(
            self.request_autoupdate
        )
        self.background_slider.sliderReleased.connect(self.request_autoupdate)
        self.fretting_slider.sliderReleased.connect(self.request_autoupdate)
        # Deriving the donor lifetime / background from layers fills the text
        # fields programmatically, which emits no ``editingFinished``.
        self.donor_lifetime_combobox.selectionChanged.connect(
            self.request_autoupdate
        )
        self.background_image_combobox.selectionChanged.connect(
            self.request_autoupdate
        )

        # NOTE: The widget is created here but NOT added to this tab's layout.
        # PlotterWidget wraps it in a HistogramDockWidget and docks it separately.
        self.histogram_widget = HistogramWidget(
            xlabel="FRET efficiency",
            ylabel="Pixel count",
            bins=150,
            default_colormap_name="viridis",
            range_slider_enabled=True,
            range_label_prefix="FRET efficiency range",
            range_factor=1000,
            viewer=self.viewer,
            parent=self,
        )

        # Connect range-slider signal
        self.histogram_widget.rangeChanged.connect(self._on_fret_range_changed)

        layout.addStretch()

        # Set the content widget as the scroll area's widget
        scroll_area.setWidget(content_widget)

        # Set the scroll area as the main layout
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(scroll_area)

        # Initialize selectors and comboboxes
        self.donor_stack.setCurrentIndex(0)
        self.bg_stack.setCurrentIndex(0)
        self._update_background_combobox()
        self._update_donor_lifetime_combobox()

    def _on_donor_source_changed(self, index: int):
        """Switch donor lifetime input mode (Manual | From layer)."""
        if hasattr(self, 'donor_stack'):
            self.donor_stack.setCurrentIndex(index)

        if index == 0:
            self.donor_label.setText("Donor lifetime (ns):")
        else:
            self.donor_label.setText("Donor lifetime (ns):")

        if (
            not hasattr(self, '_updating_settings')
            or not self._updating_settings
        ):
            source_text = 'Manual' if index == 0 else 'From layer(s)'
            self._update_fret_setting_in_metadata('donor_source', source_text)

        if index == 1:
            self._calculate_donor_lifetime()

    def _on_bg_source_changed(self, index: int):
        """Switch background position input mode (Manual - From image)."""
        if hasattr(self, 'bg_stack'):
            self.bg_stack.setCurrentIndex(index)

        if index == 0:
            self.background_position_label.setText("Background position:")
        else:
            self.background_position_label.setText("Background position:")

        if (
            not hasattr(self, '_updating_settings')
            or not self._updating_settings
        ):
            source_text = 'Manual' if index == 0 else 'From layer(s)'
            self._update_fret_setting_in_metadata(
                'background_source', source_text
            )

        if index == 1:
            self._calculate_background_position()

    def get_all_artists(self):
        """Return a list of all matplotlib artists created by this widget."""
        artists = []
        if self.current_donor_line is not None:
            artists.append(self.current_donor_line)
        if self.current_donor_circle is not None:
            artists.append(self.current_donor_circle)
        if self.current_background_circle is not None:
            artists.append(self.current_background_circle)
        return artists

    def set_artists_visible(self, visible):
        """Set visibility of all artists created by this widget."""
        for artist in self.get_all_artists():
            if hasattr(artist, 'set_visible'):
                artist.set_visible(visible)

    def clear_artists(self):
        """Clear (remove) all artists created by this widget."""
        if self.current_donor_line is not None:
            with contextlib.suppress(ValueError):
                self.current_donor_line.remove()
            self.current_donor_line = None

        if self.current_donor_circle is not None:
            with contextlib.suppress(ValueError):
                self.current_donor_circle.remove()
            self.current_donor_circle = None

        if self.current_background_circle is not None:
            with contextlib.suppress(ValueError):
                self.current_background_circle.remove()
            self.current_background_circle = None

        if self.parent_widget is not None:
            self.parent_widget.canvas_widget.canvas.draw_idle()

    def _on_harmonic_changed(self):
        """Handle harmonic changes from the parent widget."""
        if self.parent_widget is None:
            return

        new_harmonic = self.parent_widget.harmonic

        if self.current_harmonic != new_harmonic:
            # Store background position for the current harmonic before switching
            # Don't set _updating_settings here so metadata gets updated
            self._store_current_background_position()

        self.current_harmonic = new_harmonic

        self._load_background_position_for_harmonic(new_harmonic)

        self._updating_settings = True
        try:
            self.plot_donor_trajectory()
        finally:
            self._updating_settings = False

    def _store_current_background_position(self):
        """Store the current background position for the current harmonic."""
        if (
            self.background_real_edit.text()
            and self.background_imag_edit.text()
        ):
            try:
                real = float(self.background_real_edit.text().strip())
                imag = float(self.background_imag_edit.text().strip())

                harmonic_exists = (
                    self.current_harmonic
                    in self.background_positions_by_harmonic
                )

                if real == 0.0 and imag == 0.0:
                    if not harmonic_exists:
                        return
                    existing = self.background_positions_by_harmonic[
                        self.current_harmonic
                    ]
                    if existing['real'] == 0.0 and existing['imag'] == 0.0:
                        return

                self.background_positions_by_harmonic[
                    self.current_harmonic
                ] = {'real': real, 'imag': imag}

                self._update_fret_setting_in_metadata(
                    'background_positions_by_harmonic',
                    self.background_positions_by_harmonic.copy(),
                )
            except ValueError:
                pass

    def _load_background_position_for_harmonic(self, harmonic):
        """Load background position for the specified harmonic."""
        self.background_real_edit.blockSignals(True)
        self.background_imag_edit.blockSignals(True)

        try:
            if harmonic in self.background_positions_by_harmonic:
                stored = self.background_positions_by_harmonic[harmonic]
                self.background_real_edit.setText(f"{stored['real']:.3f}")
                self.background_imag_edit.setText(f"{stored['imag']:.3f}")
                self.background_real = stored['real']
                self.background_imag = stored['imag']
            else:
                self.background_real_edit.setText("0.000")
                self.background_imag_edit.setText("0.000")
                self.background_real = 0.0
                self.background_imag = 0.0
        finally:
            self.background_real_edit.blockSignals(False)
            self.background_imag_edit.blockSignals(False)

    def _on_background_position_changed(self):
        """Handle manual changes to background position fields."""
        # Don't update during settings restoration
        if hasattr(self, '_updating_settings') and self._updating_settings:
            return

        try:
            real = float(self.background_real_edit.text().strip())
            imag = float(self.background_imag_edit.text().strip())
            self.background_real = real
            self.background_imag = imag

            self.background_positions_by_harmonic[self.current_harmonic] = {
                'real': real,
                'imag': imag,
            }

            layer_name = self.parent_widget.get_primary_layer_name()
            if layer_name:
                self._update_fret_setting_in_metadata(
                    'background_positions_by_harmonic',
                    self.background_positions_by_harmonic,
                )
        except ValueError:
            show_error(
                "Invalid background position: please enter numeric values."
            )

        self._on_parameters_changed()

    def _on_background_slider_changed(self):
        """Handle background slider value change."""
        value = self.background_slider.value() / 100.0
        self.donor_background = value
        self.background_label.setText(f"{value:.2f}")

        if (
            not hasattr(self, '_updating_settings')
            or not self._updating_settings
        ):
            self._update_fret_setting_in_metadata('donor_background', value)

        self._on_parameters_changed()

    def _on_fretting_slider_changed(self):
        """Handle fretting proportion slider value change."""
        value = self.fretting_slider.value() / 100.0
        self.donor_fretting_proportion = value
        self.fretting_label.setText(f"{value:.2f}")

        if (
            not hasattr(self, '_updating_settings')
            or not self._updating_settings
        ):
            self._update_fret_setting_in_metadata(
                'donor_fretting_proportion', value
            )

        self._on_parameters_changed()

    def _on_colormap_checkbox_changed(self, checked=None):
        """Handle colormap checkbox state change."""
        self.use_colormap = self.colormap_checkbox.isChecked()

        if (
            not hasattr(self, '_updating_settings')
            or not self._updating_settings
        ):
            self._update_fret_setting_in_metadata(
                'use_colormap', self.use_colormap
            )

        self.plot_donor_trajectory()

    def _on_parameters_changed(self):
        """Update plot when any parameter changes."""
        if self._updating_settings:
            return

        layer_name = self.parent_widget.get_primary_layer_name()
        if not layer_name:
            return

        if self.donor_line_edit.text():
            try:
                self.donor_lifetime = float(self.donor_line_edit.text())
                self._update_fret_setting_in_metadata(
                    'donor_lifetime', self.donor_lifetime
                )
            except ValueError:
                show_error(
                    "Invalid donor lifetime: please enter a numeric value."
                )

        if (
            self.donor_line_edit.text()
            and self.frequency_input.text()
            and self.background_real_edit.text()
            and self.background_imag_edit.text()
        ):
            try:
                base_frequency = float(self.frequency_input.text().strip())
                self.frequency = base_frequency * self.parent_widget.harmonic
                self.donor_lifetime = float(self.donor_line_edit.text())
                self.background_real = float(self.background_real_edit.text())
                self.background_imag = float(self.background_imag_edit.text())
                self.plot_donor_trajectory()
            except ValueError:
                pass

    def _update_background_combobox(self):
        """Update the background image combobox with available layers."""
        if getattr(self, '_updating_background_combobox', False):
            return

        self._updating_background_combobox = True

        try:
            previously_checked = self.background_image_combobox.checkedItems()
            self.background_image_combobox.blockSignals(True)
            self.background_image_combobox.clear()

            layer_names = [
                layer.name
                for layer in self.viewer.layers
                if isinstance(layer, Image)
                and "G" in layer.metadata
                and "S" in layer.metadata
                and "G_original" in layer.metadata
                and "S_original" in layer.metadata
            ]

            for name in layer_names:
                checked = name in previously_checked
                self.background_image_combobox.addItem(name, checked)

            self.background_image_combobox.blockSignals(False)

            for layer in self.viewer.layers:
                if isinstance(layer, Image):
                    with contextlib.suppress(TypeError, ValueError):
                        layer.events.name.disconnect(
                            self._update_background_combobox
                        )
                    layer.events.name.connect(self._update_background_combobox)
                    self._name_event_layers[id(layer)] = layer

        finally:
            self._updating_background_combobox = False

    def _update_donor_lifetime_combobox(self):
        """Update the donor lifetime combobox with available layers."""
        if getattr(self, '_updating_donor_combobox', False):
            return

        self._updating_donor_combobox = True

        try:
            previously_checked = self.donor_lifetime_combobox.checkedItems()
            self.donor_lifetime_combobox.blockSignals(True)
            self.donor_lifetime_combobox.clear()

            layer_names = [
                layer.name
                for layer in self.viewer.layers
                if isinstance(layer, Image)
                and "G" in layer.metadata
                and "S" in layer.metadata
                and "G_original" in layer.metadata
                and "S_original" in layer.metadata
            ]

            for name in layer_names:
                checked = name in previously_checked
                self.donor_lifetime_combobox.addItem(name, checked)

            self.donor_lifetime_combobox.blockSignals(False)

            for layer in self.viewer.layers:
                if isinstance(layer, Image):
                    with contextlib.suppress(TypeError, ValueError):
                        layer.events.name.disconnect(
                            self._update_donor_lifetime_combobox
                        )
                    layer.events.name.connect(
                        self._update_donor_lifetime_combobox
                    )
                    self._name_event_layers[id(layer)] = layer

        finally:
            self._updating_donor_combobox = False

    def _on_layer_changed(self):
        """Handle when layers are added or removed in the viewer."""
        current_ids = {id(layer) for layer in self.viewer.layers}
        for layer_id, layer in list(self._name_event_layers.items()):
            if layer_id in current_ids:
                continue
            with contextlib.suppress(TypeError, ValueError):
                layer.events.name.disconnect(self._update_background_combobox)
            with contextlib.suppress(TypeError, ValueError):
                layer.events.name.disconnect(
                    self._update_donor_lifetime_combobox
                )
            self._name_event_layers.pop(layer_id, None)
        self._update_background_combobox()
        self._update_donor_lifetime_combobox()

    def _calculate_background_position(self):
        """Calculate the background position from selected background image layers for all harmonics.

        When multiple layers are selected, the phasor center is computed for each layer
        and the results are averaged across all selected layers.
        """
        selected_layer_names = self.background_image_combobox.checkedItems()
        if not selected_layer_names:
            if self.bg_source_selector.currentIndex() == 1:
                self.background_position_label.setText("Background position:")
            return

        if (
            not hasattr(self, '_updating_settings')
            or not self._updating_settings
        ):
            self._update_fret_setting_in_metadata(
                'background_layer_names', selected_layer_names
            )

        # Collect phasor centers per harmonic across all selected layers
        positions_by_harmonic = {}  # harmonic -> list of (real, imag)

        for background_layer_name in selected_layer_names:
            try:
                background_layer = self.viewer.layers[background_layer_name]
                g_array = background_layer.metadata.get("G")
                s_array = background_layer.metadata.get("S")
                harmonics = background_layer.metadata.get("harmonics")
                mean = background_layer.metadata.get("original_mean")

                if g_array is None or s_array is None:
                    continue

            except (KeyError, AttributeError):
                continue

            if harmonics is not None:
                harmonics_arr = np.atleast_1d(harmonics)
                for harmonic in harmonics_arr:
                    try:
                        harmonic_idx = np.where(harmonics_arr == harmonic)[0]
                        if len(harmonic_idx) == 0:
                            continue
                        harmonic_idx = harmonic_idx[0]

                        if g_array.ndim > background_layer.data.ndim:
                            real = g_array[harmonic_idx]
                            imag = s_array[harmonic_idx]
                        else:
                            real = g_array
                            imag = s_array

                        _, center_real, center_imag = phasor_center(
                            mean, real, imag
                        )

                        if harmonic not in positions_by_harmonic:
                            positions_by_harmonic[harmonic] = []
                        positions_by_harmonic[harmonic].append(
                            (center_real, center_imag)
                        )
                    except Exception:  # noqa: BLE001
                        continue
            else:
                try:
                    _, center_real, center_imag = phasor_center(
                        mean, g_array, s_array
                    )
                    harmonic = self.parent_widget.harmonic
                    if harmonic not in positions_by_harmonic:
                        positions_by_harmonic[harmonic] = []
                    positions_by_harmonic[harmonic].append(
                        (center_real, center_imag)
                    )
                except Exception:  # noqa: BLE001
                    continue

        if not positions_by_harmonic:
            if self.bg_source_selector.currentIndex() == 1:
                self.background_position_label.setText("Background position:")
            return

        # Average positions across layers for each harmonic
        for harmonic, coords in positions_by_harmonic.items():
            avg_real = float(np.mean([c[0] for c in coords]))
            avg_imag = float(np.mean([c[1] for c in coords]))
            self.background_positions_by_harmonic[harmonic] = {
                'real': avg_real,
                'imag': avg_imag,
            }

        if (
            not hasattr(self, '_updating_settings')
            or not self._updating_settings
        ):
            self._update_fret_setting_in_metadata(
                'background_positions_by_harmonic',
                self.background_positions_by_harmonic,
            )

        current_harmonic = self.parent_widget.harmonic
        if current_harmonic in self.background_positions_by_harmonic:
            stored = self.background_positions_by_harmonic[current_harmonic]
            self.background_real_edit.setText(f"{stored['real']:.3f}")
            self.background_imag_edit.setText(f"{stored['imag']:.3f}")
            self.background_real = stored['real']
            self.background_imag = stored['imag']

            if self.bg_source_selector.currentIndex() == 1:
                self.background_position_label.setText(
                    f"Background position: G={stored['real']:.2f}, S={stored['imag']:.2f}"
                )
        else:
            self.background_real_edit.setText("0.00")
            self.background_imag_edit.setText("0.00")
            self.background_real = 0.0
            self.background_imag = 0.0

            if self.bg_source_selector.currentIndex() == 1:
                self.background_position_label.setText(
                    "Background position: G=0.00, S=0.00"
                )

        self.plot_donor_trajectory()

    def _calculate_donor_lifetime(self):
        """Calculate the donor lifetime from selected layers for current harmonic.

        When multiple layers are selected, the phasor center is computed for each layer
        and the resulting lifetimes are averaged.
        """
        selected_layer_names = self.donor_lifetime_combobox.checkedItems()
        if not selected_layer_names:
            if self.donor_source_selector.currentIndex() == 1:
                self.donor_label.setText("Donor lifetime (ns):")
            return

        if (
            not hasattr(self, '_updating_settings')
            or not self._updating_settings
        ):
            self._update_fret_setting_in_metadata(
                'donor_layer_names', selected_layer_names
            )

        if not self.frequency_input.text():
            if self.donor_source_selector.currentIndex() == 1:
                self.donor_label.setText("Donor lifetime (ns):")
            return

        try:
            frequency = float(self.frequency_input.text().strip())
        except ValueError:
            return

        current_harmonic = self.parent_widget.harmonic
        lifetime_type = self.lifetime_type_combobox.currentText()

        if (
            not hasattr(self, '_updating_settings')
            or not self._updating_settings
        ):
            self._update_fret_setting_in_metadata(
                'donor_lifetime_type', lifetime_type
            )

        lifetimes = []
        for donor_layer_name in selected_layer_names:
            try:
                donor_layer = self.viewer.layers[donor_layer_name]
                g_array = donor_layer.metadata.get("G")
                s_array = donor_layer.metadata.get("S")
                harmonics = donor_layer.metadata.get("harmonics")
                mean = donor_layer.metadata.get("original_mean")

                if g_array is None or s_array is None:
                    continue

                if harmonics is not None:
                    harmonics = np.atleast_1d(harmonics)
                    harmonic_idx = np.where(harmonics == current_harmonic)[0]

                    if len(harmonic_idx) == 0:
                        continue

                    harmonic_idx = harmonic_idx[0]

                    if g_array.ndim > donor_layer.data.ndim:
                        real = g_array[harmonic_idx]
                        imag = s_array[harmonic_idx]
                    else:
                        real = g_array
                        imag = s_array
                else:
                    real = g_array
                    imag = s_array

                _, center_real, center_imag = phasor_center(mean, real, imag)

                if lifetime_type in (
                    "Apparent Phase Lifetime",
                    "Apparent Modulation Lifetime",
                ):
                    phase_lifetime, mod_lifetime = phasor_to_apparent_lifetime(
                        center_real, center_imag, frequency=frequency
                    )
                    lifetime = {
                        "Apparent Phase Lifetime": phase_lifetime,
                        "Apparent Modulation Lifetime": mod_lifetime,
                    }[lifetime_type]
                elif lifetime_type == "Normal Lifetime":
                    lifetime = phasor_to_normal_lifetime(
                        center_real, center_imag, frequency=frequency
                    )
                else:
                    continue

                lifetimes.append(float(lifetime))

            except Exception:  # noqa: BLE001
                continue

        if not lifetimes:
            if self.donor_source_selector.currentIndex() == 1:
                self.donor_label.setText("Donor lifetime (ns):")
            return

        avg_lifetime = float(np.mean(lifetimes))
        self.donor_line_edit.setText(f"{avg_lifetime:.2f}")
        self.donor_lifetime = avg_lifetime

        if (
            not hasattr(self, '_updating_settings')
            or not self._updating_settings
        ):
            self._update_fret_setting_in_metadata(
                'donor_lifetime', avg_lifetime
            )

        if self.donor_source_selector.currentIndex() == 1:
            self.donor_label.setText(
                f"Donor lifetime (from layer(s)): {avg_lifetime:.2f} ns"
            )

        self.plot_donor_trajectory()

    def plot_donor_trajectory(self):
        """Plot the donor trajectory with current parameters."""
        try:
            if self.current_donor_line is not None:
                with contextlib.suppress(ValueError):
                    self.current_donor_line.remove()
                self.current_donor_line = None

            if self.current_donor_circle is not None:
                with contextlib.suppress(ValueError):
                    self.current_donor_circle.remove()
                self.current_donor_circle = None

            if self.current_background_circle is not None:
                with contextlib.suppress(ValueError):
                    self.current_background_circle.remove()
                self.current_background_circle = None

            if not (
                self.donor_line_edit.text() and self.frequency_input.text()
            ):
                return

            base_frequency = float(self.frequency_input.text().strip())
            self.frequency = base_frequency * self.parent_widget.harmonic
            self.donor_lifetime = float(self.donor_line_edit.text().strip())
            self.background_real = float(
                self.background_real_edit.text().strip()
            )
            self.background_imag = float(
                self.background_imag_edit.text().strip()
            )

            donor_trajectory_real, donor_trajectory_imag = (
                phasor_from_fret_donor(
                    self.frequency,
                    self.donor_lifetime,
                    fret_efficiency=self._fret_efficiencies,
                    donor_background=self.donor_background,
                    background_imag=self.background_imag,
                    background_real=self.background_real,
                    donor_fretting=self.donor_fretting_proportion,
                )
            )

            ax = self.parent_widget.canvas_widget.figure.gca()

            if self.fret_layer is not None and self.use_colormap:
                if (
                    hasattr(self, 'fret_colormap')
                    and self.fret_colormap is not None
                ):
                    colormap = LinearSegmentedColormap.from_list(
                        "fret_interp", self.fret_colormap, N=256
                    )
                else:
                    colormap = plt.cm.jet

                donor_color = colormap(0.0)[:3]
                background_color = colormap(1.0)[:3]
            else:
                donor_color = 'dimgray'
                background_color = 'dimgray'

            trajectory_zorder = 10
            dot_zorder = 11

            if self.fret_layer is not None and self.use_colormap:
                self._draw_colormap_trajectory(
                    ax,
                    donor_trajectory_real,
                    donor_trajectory_imag,
                    zorder=trajectory_zorder,
                )
            else:
                self.current_donor_line = ax.plot(
                    donor_trajectory_real,
                    donor_trajectory_imag,
                    color='dimgray',
                    linewidth=3,
                    label='Donor Trajectory',
                    zorder=trajectory_zorder,
                )[0]

            circle_radius = 0.02

            donor_circle = plt.Circle(
                (donor_trajectory_real[0], donor_trajectory_imag[0]),
                circle_radius,
                fill=True,
                facecolor=donor_color,
                linewidth=1,
                zorder=dot_zorder,
            )
            self.current_donor_circle = ax.add_patch(donor_circle)

            background_circle = plt.Circle(
                (donor_trajectory_real[-1], donor_trajectory_imag[-1]),
                circle_radius,
                fill=True,
                facecolor=background_color,
                linewidth=1,
                zorder=dot_zorder,
            )
            self.current_background_circle = ax.add_patch(background_circle)

            self.parent_widget.canvas_widget.canvas.draw_idle()

        except Exception as e:  # noqa: BLE001
            show_error(f"Error drawing line: {str(e)}")

    def _draw_colormap_trajectory(
        self, ax, trajectory_real, trajectory_imag, zorder=10
    ):
        """Draw a colormap trajectory line."""
        num_segments = min(
            len(trajectory_real) * self.colormap_density_factor,
            len(trajectory_real) - 1,
        )

        if hasattr(self, 'fret_colormap') and self.fret_colormap is not None:
            colormap = LinearSegmentedColormap.from_list(
                "fret_interp", self.fret_colormap, N=256
            )
        else:
            colormap = plt.cm.jet

        if (
            hasattr(self, 'colormap_contrast_limits')
            and self.colormap_contrast_limits is not None
        ):
            vmin, vmax = self.colormap_contrast_limits
        elif self.fret_layer is not None:
            vmin, vmax = self.fret_layer.contrast_limits
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

            fret_value = self._fret_efficiencies[start_idx]
            colors.append(fret_value)

        lc = LineCollection(
            segments, cmap=colormap, linewidths=3, zorder=zorder
        )
        lc.set_array(np.array(colors))
        gamma = getattr(self, 'colormap_gamma', 1.0) or 1.0
        if gamma != 1.0 and vmax > vmin:
            # Match the FRET layer's gamma so the phasor-plot trajectory reads
            # the same as the image and the histogram.
            lc.set_norm(PowerNorm(gamma, vmin=vmin, vmax=vmax))
        else:
            lc.set_clim(vmin, vmax)

        self.current_donor_line = ax.add_collection(lc)

    def _on_colormap_changed(self, event):
        """Handle changes to the colormap of any FRET layer and sync all layers."""
        if self._updating_linked_layers:
            return

        source_layer = event.source
        new_colormap = source_layer.colormap
        new_gamma = source_layer.gamma

        # Update stored values
        self.fret_colormap = new_colormap.colors
        self.colormap_contrast_limits = source_layer.contrast_limits
        self.colormap_gamma = new_gamma

        # Update histogram colormap
        self.histogram_widget.update_colormap(
            colormap_colors=self.fret_colormap,
            contrast_limits=list(self.colormap_contrast_limits),
            gamma=self.colormap_gamma,
        )

        # Extract colormap info for metadata
        colormap_name = getattr(new_colormap, 'name', 'custom')
        colormap_colors = getattr(new_colormap, 'colors', None)

        if colormap_colors is not None and (
            hasattr(colormap_colors, 'tolist')
            or isinstance(colormap_colors, np.ndarray)
        ):
            colormap_colors = colormap_colors.tolist()

        self._update_fret_setting_in_metadata(
            'colormap_settings.colormap_name', colormap_name
        )
        self._update_fret_setting_in_metadata(
            'colormap_settings.colormap_colors', colormap_colors
        )
        self._update_fret_setting_in_metadata(
            'colormap_settings.colormap_changed', True
        )
        self._update_fret_setting_in_metadata(
            'colormap_settings.gamma', new_gamma
        )

        # Update all other FRET layers to match
        self._updating_linked_layers = True
        try:
            for layer in self.fret_layers:
                if layer != source_layer and layer in self.viewer.layers:
                    layer.colormap = new_colormap
                    layer.gamma = new_gamma
        finally:
            self._updating_linked_layers = False

        self._remember_fret_display(source_layer)
        self.plot_donor_trajectory()

    def _on_contrast_limits_changed(self, event):
        """Handle changes to the contrast limits of any FRET layer and sync all layers."""
        if self._updating_linked_layers:
            return

        source_layer = event.source
        new_contrast_limits = source_layer.contrast_limits

        # Update stored values
        self.colormap_contrast_limits = new_contrast_limits

        # Update histogram colormap
        self.histogram_widget.update_colormap(
            colormap_colors=self.fret_colormap,
            contrast_limits=list(self.colormap_contrast_limits),
            gamma=self.colormap_gamma,
        )

        # Prepare for metadata
        contrast_limits = new_contrast_limits
        if hasattr(contrast_limits, 'tolist') or isinstance(
            contrast_limits, np.ndarray
        ):
            contrast_limits = contrast_limits.tolist()

        self._update_fret_setting_in_metadata(
            'colormap_settings.contrast_limits', contrast_limits
        )

        # Update all other FRET layers to match
        self._updating_linked_layers = True
        try:
            for layer in self.fret_layers:
                if layer != source_layer and layer in self.viewer.layers:
                    layer.contrast_limits = new_contrast_limits
        finally:
            self._updating_linked_layers = False

        self._remember_fret_display(source_layer)
        self.plot_donor_trajectory()

    def _remember_fret_display(self, layer):
        """Keep *layer*'s colormap, limits and gamma for the next run.

        The saved display is what a new run applies to the FRET layers. It
        is otherwise only read from the metadata when the layer selection
        changes, so without this a run would bring back whatever colormap
        was saved then, undoing the user's later changes.
        """
        colormap = layer.colormap
        colors = getattr(colormap, 'colors', None)
        self._saved_colormap_name = getattr(colormap, 'name', 'custom')
        self._saved_colormap_colors = (
            None if colors is None else np.asarray(colors).tolist()
        )
        self._saved_contrast_limits = [float(v) for v in layer.contrast_limits]
        self._saved_gamma = layer.gamma

    def _saved_fret_colormap(self):
        """Return the saved colormap as a layer colormap value."""
        colormap = layer_colormap_from_settings(
            {
                'colormap_name': self._saved_colormap_name,
                'colormap_colors': self._saved_colormap_colors,
            }
        )
        return colormap if colormap is not None else 'viridis'

    def _get_default_fret_settings(self):
        """Get default settings dictionary for FRET parameters."""
        return {
            'donor_lifetime': None,
            'donor_background': 0.1,
            'donor_fretting_proportion': 1.0,
            'use_colormap': True,
            'background_positions_by_harmonic': {},
            'colormap_settings': {
                'colormap_name': 'viridis',
                'colormap_colors': None,
                'contrast_limits': (0, 1),
                'colormap_changed': False,
            },
            'donor_source': 'Manual',
            'donor_layer_names': [],
            'donor_lifetime_type': 'Apparent Phase Lifetime',
            'background_source': 'Manual',
            'background_layer_names': [],
        }

    def _has_settings_store(self):
        """Return whether the parent keeps per-layer settings."""
        return self.parent_widget is not None and hasattr(
            self.parent_widget, 'settings_store'
        )

    def _update_fret_setting_in_metadata(self, key_path, value):
        """Keep an edited FRET setting as the primary's unsaved setting.

        It is stored in the layers when Calculate runs (see
        :meth:`_commit_fret_settings`). Colormap settings describe the FRET
        layers already on display instead, so they are stored right away in
        every layer that has one.
        """
        if self._updating_settings or not self._has_settings_store():
            return

        path = tuple(key_path.split('.'))
        store = self.parent_widget.settings_store
        if path[0] == 'colormap_settings':
            layers = self._fret_analysed_layers()
            if layers:
                store.update_committed(layers, 'fret', path, value)
            return

        primary = self.parent_widget.get_primary_layer()
        if primary is None:
            return
        store.set_draft_path(
            primary,
            'fret',
            path,
            value,
            default_factory=self._get_default_fret_settings,
        )

    def _fret_analysed_layers(self):
        """Return the source layers of the FRET layers on display."""
        return [
            self.viewer.layers[name]
            for name in self._fret_output_layers()
            if name in self.viewer.layers
        ]

    def _collect_fret_settings(self):
        """Return the FRET settings a Calculate would store.

        The primary layer's settings (with its unsaved edits) completed by
        what the controls show. A donor lifetime or background position read
        from other layers is stored as its value, so the analysis can be
        repeated without those layers.
        """
        block = self._get_default_fret_settings()
        primary = (
            self.parent_widget.get_primary_layer()
            if self._has_settings_store()
            else None
        )
        if primary is not None:
            stored = self.parent_widget.layer_settings(primary).get('fret')
            if isinstance(stored, dict):
                block.update(copy.deepcopy(stored))
        try:
            donor_lifetime = float(self.donor_line_edit.text())
        except ValueError:
            donor_lifetime = block.get('donor_lifetime')
        manual_donor = self.donor_source_selector.currentIndex() == 0
        manual_background = self.bg_source_selector.currentIndex() == 0
        block.update(
            {
                'donor_lifetime': donor_lifetime,
                'donor_background': self.donor_background,
                'donor_fretting_proportion': self.donor_fretting_proportion,
                'use_colormap': self.use_colormap,
                'background_positions_by_harmonic': copy.deepcopy(
                    self.background_positions_by_harmonic
                ),
                'donor_source': 'Manual' if manual_donor else 'From layer(s)',
                'donor_layer_names': list(
                    self.donor_lifetime_combobox.checkedItems()
                ),
                'donor_lifetime_type': (
                    self.lifetime_type_combobox.currentText()
                ),
                'background_source': (
                    'Manual' if manual_background else 'From layer(s)'
                ),
                'background_layer_names': list(
                    self.background_image_combobox.checkedItems()
                ),
            }
        )
        return block

    def _fret_merge_rule(self):
        """Return how a run merges into other layers' stored settings.

        Background positions are kept per harmonic; a run only replaces the
        one of the harmonic it used.
        """
        harmonic = getattr(self.parent_widget, 'harmonic', 1)
        return {
            'fret': merge_keyed_path(
                ('background_positions_by_harmonic',), [harmonic]
            )
        }

    def _commit_fret_settings(self, layers):
        """Store the run's FRET settings in every analysed layer."""
        if not layers or not self._has_settings_store():
            return
        self.parent_widget.commit_analysis_settings(
            {'fret': self._collect_fret_settings()},
            layers=layers,
            merge=self._fret_merge_rule(),
        )

    def _refresh_settings_note(self):
        """Caution about settings and frequencies a Calculate would change."""
        note = getattr(self, '_settings_note', None)
        if note is None or not self._has_settings_store():
            return
        if self._needs_update:
            # The controls still show another layer; refreshed on restore.
            return
        messages = [
            self.parent_widget.settings_overwrite_message(
                'fret_tab',
                values={'fret': self._collect_fret_settings()},
                merge=self._fret_merge_rule(),
                action="Calculating",
            )
        ]
        messages += self.parent_widget.frequency_note_messages(
            self.frequency_input.text()
        )
        set_settings_note(note, messages)

    def _restore_fret_settings_from_metadata(self):
        """Restore all FRET settings from the current layer's metadata."""
        layer_name = self.parent_widget.get_primary_layer_name()
        if not layer_name or layer_name not in self.viewer.layers:
            self.background_positions_by_harmonic = {}
            return

        layer = self.viewer.layers[layer_name]
        # Includes the unsaved edits made while the layer was primary.
        layer_settings = self.parent_widget.layer_settings(layer)
        if not isinstance(layer_settings.get('fret'), dict):
            self.background_positions_by_harmonic = {}
            return

        self._updating_settings = True
        try:
            settings = layer_settings['fret']

            # Initialize or restore background positions by harmonic
            if settings.get('background_positions_by_harmonic'):
                # Convert string keys to integers (they get stringified in JSON)
                bg_positions = settings['background_positions_by_harmonic']
                self.background_positions_by_harmonic = {
                    int(k) if isinstance(k, str) and k.isdigit() else k: v
                    for k, v in bg_positions.items()
                }
            else:
                self.background_positions_by_harmonic = {}

            self.donor_line_edit.clear()
            self.frequency_input.clear()

            if settings.get('donor_lifetime') is not None:
                self.donor_lifetime = settings['donor_lifetime']
                self.donor_line_edit.setText(str(self.donor_lifetime))

            frequency = layer_settings.get('frequency')
            if frequency is not None:
                self.frequency_input.setText(str(frequency))
                self.frequency = frequency * self.parent_widget.harmonic
            else:
                self.frequency_input.setText("")
                self.frequency = None

            if settings.get('donor_background') is not None:
                self.donor_background = settings['donor_background']
                self.background_slider.setValue(
                    int(self.donor_background * 100)
                )
                self.background_label.setText(f"{self.donor_background:.2f}")

            self._load_background_position_for_harmonic(self.current_harmonic)

            if settings.get('donor_fretting_proportion') is not None:
                self.donor_fretting_proportion = settings[
                    'donor_fretting_proportion'
                ]
                self.fretting_slider.setValue(
                    int(self.donor_fretting_proportion * 100)
                )
                self.fretting_label.setText(
                    f"{self.donor_fretting_proportion:.2f}"
                )

            if settings.get('use_colormap') is not None:
                self.use_colormap = settings['use_colormap']
                self.colormap_checkbox.setChecked(self.use_colormap)

            donor_source = settings.get('donor_source', 'Manual')
            # Support both old single-name format and new list format
            donor_layer_names_raw = settings.get('donor_layer_names', None)
            if donor_layer_names_raw is None:
                old_name = settings.get('donor_layer_name', None)
                donor_layer_names_raw = [old_name] if old_name else []
            donor_layer_names = [
                n for n in donor_layer_names_raw if n in self.viewer.layers
            ]
            donor_lifetime_type = settings.get(
                'donor_lifetime_type', 'Apparent Phase Lifetime'
            )

            if (
                donor_source in ('From layer', 'From layer(s)')
                and donor_layer_names
            ):
                self.donor_source_selector.setCurrentIndex(1)
                self.donor_lifetime_combobox.setCheckedItems(donor_layer_names)
                type_index = self.lifetime_type_combobox.findText(
                    donor_lifetime_type
                )
                if type_index >= 0:
                    self.lifetime_type_combobox.setCurrentIndex(type_index)
            else:
                self.donor_source_selector.setCurrentIndex(0)
                if settings.get('donor_lifetime') is not None:
                    self.donor_line_edit.setText(
                        str(settings['donor_lifetime'])
                    )

            background_source = settings.get('background_source', 'Manual')
            # Support both old single-name format and new list format
            background_layer_names_raw = settings.get(
                'background_layer_names', None
            )
            if background_layer_names_raw is None:
                old_name = settings.get('background_layer_name', None)
                background_layer_names_raw = [old_name] if old_name else []
            background_layer_names = [
                n
                for n in background_layer_names_raw
                if n in self.viewer.layers
            ]

            if (
                background_source in ('From layer', 'From layer(s)')
                and background_layer_names
            ):
                self.bg_source_selector.setCurrentIndex(1)
                self.background_image_combobox.setCheckedItems(
                    background_layer_names
                )
            else:
                self.bg_source_selector.setCurrentIndex(0)

            if 'colormap_settings' in settings:
                colormap_settings = settings['colormap_settings']
                self._saved_colormap_name = colormap_settings.get(
                    'colormap_name', 'viridis'
                )
                self._saved_colormap_colors = colormap_settings.get(
                    'colormap_colors', None
                )
                self._saved_contrast_limits = colormap_settings.get(
                    'contrast_limits', (0, 1)
                )
                self._saved_gamma = colormap_settings.get('gamma', 1.0)
                self._colormap_was_changed = colormap_settings.get(
                    'colormap_changed', False
                )

        finally:
            self._updating_settings = False

    def _recreate_fret_from_metadata(self):
        """Recreate FRET analysis from metadata if it was previously performed."""
        layer_name = self.parent_widget.get_primary_layer_name()
        if layer_name and layer_name in self.viewer.layers:
            layer = self.viewer.layers[layer_name]
            if 'fret' in self.parent_widget.layer_settings(layer):

                if self.donor_line_edit.text() and self.frequency_input.text():
                    self._updating_settings = True
                    try:
                        self.calculate_fret_efficiency()
                        if hasattr(self, '_saved_colormap_name'):
                            self._apply_saved_fret_colormap_settings()
                    finally:
                        self._updating_settings = False
            else:
                if self.donor_line_edit.text() and self.frequency_input.text():
                    self.plot_donor_trajectory()

    def _apply_saved_fret_colormap_settings(self):
        """Apply saved colormap settings to FRET layer if it exists."""
        if self.fret_layer is not None and hasattr(
            self, '_saved_colormap_name'
        ):

            try:
                self.fret_layer.events.colormap.disconnect(
                    self._on_colormap_changed
                )
                self.fret_layer.events.contrast_limits.disconnect(
                    self._on_contrast_limits_changed
                )
                self.fret_layer.events.gamma.disconnect(
                    self._on_colormap_changed
                )

                self.fret_layer.colormap = self._saved_fret_colormap()

                if isinstance(self._saved_contrast_limits, list):
                    saved_limits = tuple(self._saved_contrast_limits)
                else:
                    saved_limits = self._saved_contrast_limits

                self.fret_layer.contrast_limits = saved_limits

                if getattr(self, '_saved_gamma', None) is not None:
                    self.fret_layer.gamma = self._saved_gamma

                self.fret_colormap = self.fret_layer.colormap.colors
                self.colormap_contrast_limits = self.fret_layer.contrast_limits
                self.colormap_gamma = self.fret_layer.gamma

                self.fret_layer.events.colormap.connect(
                    self._on_colormap_changed
                )
                self.fret_layer.events.contrast_limits.connect(
                    self._on_contrast_limits_changed
                )
                self.fret_layer.events.gamma.connect(self._on_colormap_changed)

                self.plot_donor_trajectory()

            except Exception as e:  # noqa: BLE001
                print(f"Error applying saved colormap settings: {e}")
                try:
                    self.fret_layer.events.colormap.connect(
                        self._on_colormap_changed
                    )
                    self.fret_layer.events.contrast_limits.connect(
                        self._on_contrast_limits_changed
                    )
                    self.fret_layer.events.gamma.connect(
                        self._on_colormap_changed
                    )
                except Exception:  # noqa: BLE001
                    pass

    def _reconnect_existing_fret_layer(self, layer_name):
        """Reconnect to existing FRET layer if it exists for this layer."""
        outputs = self._fret_output_layers()
        self._set_fret_layers(outputs.values())
        self.fret_layer = outputs.get(layer_name)
        if self.fret_layer is None:
            return

        if hasattr(self, '_saved_colormap_name'):
            self._apply_saved_fret_colormap_settings()
        else:
            self.fret_colormap = self.fret_layer.colormap.colors
            self.colormap_contrast_limits = self.fret_layer.contrast_limits
            self.colormap_gamma = self.fret_layer.gamma

    def _get_selected_source_names(self) -> set[str]:
        """Return names checked in the plotter's Phasor Layers selector."""
        if self.parent_widget is None:
            return set()
        try:
            return {
                layer.name
                for layer in self.parent_widget.get_selected_layers()
            }
        except (AttributeError, RuntimeError):
            return set()

    def _fret_output_source(self, layer):
        """Return the source name for a FRET output layer."""
        if not isinstance(layer, Image):
            return None
        tag = layer.metadata.get(_FRET_OUTPUT_METADATA_KEY)
        if isinstance(tag, dict) and tag.get('source_layer'):
            return tag['source_layer']
        prefix = "FRET efficiency: "
        if layer.name.startswith(prefix):
            source_name = layer.name[len(prefix) :]
            if source_name in self.viewer.layers:
                source_layer = self.viewer.layers[source_name]
                if (
                    isinstance(source_layer, Image)
                    and 'G' in source_layer.metadata
                    and 'S' in source_layer.metadata
                ):
                    return source_name
        return None

    def _fret_output_layers(self, selected_only=False):
        """Return FRET output layers keyed by source name."""
        selected_names = (
            self._get_selected_source_names() if selected_only else None
        )
        result = {}
        for layer in self.viewer.layers:
            source_name = self._fret_output_source(layer)
            if source_name is None:
                continue
            if (
                selected_names is not None
                and source_name not in selected_names
            ):
                continue
            existing = result.get(source_name)
            layer_is_tagged = isinstance(
                layer.metadata.get(_FRET_OUTPUT_METADATA_KEY), dict
            )
            existing_is_tagged = existing is not None and isinstance(
                existing.metadata.get(_FRET_OUTPUT_METADATA_KEY), dict
            )
            if existing is None or (
                layer_is_tagged and not existing_is_tagged
            ):
                result[source_name] = layer
        return result

    def _set_fret_layers(self, layers):
        """Replace the FRET layer registry without duplicate event handlers."""
        for layer in self.fret_layers:
            with contextlib.suppress(
                AttributeError, RuntimeError, TypeError, ValueError
            ):
                layer.events.colormap.disconnect(self._on_colormap_changed)
                layer.events.contrast_limits.disconnect(
                    self._on_contrast_limits_changed
                )
                layer.events.gamma.disconnect(self._on_colormap_changed)

        self.fret_layers = list(layers)
        for layer in self.fret_layers:
            with contextlib.suppress(
                AttributeError, RuntimeError, TypeError, ValueError
            ):
                layer.events.colormap.disconnect(self._on_colormap_changed)
                layer.events.contrast_limits.disconnect(
                    self._on_contrast_limits_changed
                )
                layer.events.gamma.disconnect(self._on_colormap_changed)
            layer.events.colormap.connect(self._on_colormap_changed)
            layer.events.contrast_limits.connect(
                self._on_contrast_limits_changed
            )
            layer.events.gamma.connect(self._on_colormap_changed)

        selected_names = self._get_selected_source_names()
        selected_layers = [
            layer
            for layer in self.fret_layers
            if self._fret_output_source(layer) in selected_names
        ]
        self.fret_layer = (
            selected_layers[0]
            if selected_layers
            else (self.fret_layers[0] if self.fret_layers else None)
        )
        if self.fret_layer is not None:
            self.fret_colormap = self.fret_layer.colormap.colors
            self.colormap_contrast_limits = self.fret_layer.contrast_limits
            self.colormap_gamma = self.fret_layer.gamma

    def _sync_fret_output_visibility(self):
        """Show FRET outputs only when their source layer is selected."""
        selected_names = self._get_selected_source_names()
        self._updating_linked_layers = True
        try:
            for source_name, layer in self._fret_output_layers().items():
                desired_visible = source_name in selected_names
                if layer.visible != desired_visible:
                    layer.visible = desired_visible
        finally:
            self._updating_linked_layers = False

    def on_layer_selection_changed(self):
        """Refresh FRET outputs after the Phasor Layers selection changes."""
        all_layers = list(self._fret_output_layers().values())
        self._set_fret_layers(all_layers)
        self._sync_fret_output_visibility()
        selected_layers = list(
            self._fret_output_layers(selected_only=True).values()
        )
        self._update_fret_histogram(
            preserve_range=self._fret_range_initialized
        )
        if selected_layers:
            range_min, range_max = self.histogram_widget.get_range()
            self._apply_fret_range_to_layers(
                selected_layers, range_min, range_max
            )
            self._update_fret_histogram(update_bounds=False)

    def rename_layer(self, old_name: str, new_name: str):
        """Rename derived layers when base layer is renamed."""
        for output_layer in list(self.viewer.layers):
            tag = output_layer.metadata.get(_FRET_OUTPUT_METADATA_KEY)
            is_tagged_match = (
                isinstance(tag, dict) and tag.get('source_layer') == old_name
            )
            is_legacy_match = (
                output_layer.name == f"FRET efficiency: {old_name}"
            )
            if not is_tagged_match and not is_legacy_match:
                continue
            old_output_name = output_layer.name
            output_layer.metadata[_FRET_OUTPUT_METADATA_KEY] = {
                'source_layer': new_name
            }
            if old_output_name == f"FRET efficiency: {old_name}":
                output_layer.name = f"FRET efficiency: {new_name}"
            if output_layer.name != old_output_name:
                self.histogram_widget.rename_dataset(
                    old_output_name, output_layer.name
                )

    def _update_fret_histogram(
        self, *, update_bounds=True, preserve_range=False
    ):
        """Update the FRET efficiency histogram from all selected FRET layers."""
        output_layers = self._fret_output_layers(selected_only=True)
        selected_layers = list(output_layers.values())
        if not selected_layers:
            self.histogram_widget.clear()
            self.histogram_widget.show()
            return

        per_layer = {layer.name: layer.data for layer in selected_layers}

        self.histogram_widget.set_dataset_sources(
            {layer.name: source for source, layer in output_layers.items()}
        )
        original_arrays = [
            np.asarray(
                layer.metadata.get('fret_data_original', layer.data)
            ).ravel()
            for layer in selected_layers
        ]
        merged = np.concatenate(
            [np.asarray(layer.data).ravel() for layer in selected_layers]
        )
        per_layer = self._slice_datasets_for_frame(per_layer)

        original_merged = np.concatenate(original_arrays)
        valid = original_merged[
            ~np.isnan(original_merged) & np.isfinite(original_merged)
        ]
        if update_bounds and len(valid) > 0:
            data_min = float(np.min(valid))
            data_max = float(np.max(valid))
            if data_max <= data_min:
                data_max = data_min + 0.01
            if preserve_range:
                old_min, old_max = self.histogram_widget.get_range()
                if old_max <= data_min or old_min >= data_max:
                    range_min, range_max = data_min, data_max
                else:
                    range_min = max(data_min, old_min)
                    range_max = min(data_max, old_max)
            else:
                range_min, range_max = data_min, data_max
            self.histogram_widget.set_range(
                range_min,
                range_max,
                slider_min=data_min,
                slider_max=data_max,
            )
            self._fret_range_initialized = True

        self.histogram_widget.update_colormap(
            colormap_colors=self.fret_colormap,
            contrast_limits=(
                list(self.colormap_contrast_limits)
                if self.colormap_contrast_limits is not None
                else None
            ),
            gamma=self.colormap_gamma,
        )
        if len(per_layer) > 1:
            self.histogram_widget.update_multi_data(per_layer)
        else:
            label, data = next(iter(per_layer.items()), ("Layer", merged))
            self.histogram_widget.update_data(data, label=label)
        self.histogram_widget.show()

    def _frame_context(self):
        """Return the plotter's time-lapse frame context, if available."""
        return getattr(self.parent_widget, 'frame_context', None)

    def _slice_datasets_for_frame(self, datasets):
        """Restrict per-layer datasets to the displayed time-lapse frame.

        The un-sliced arrays are handed to the histogram widget as well, so
        the statistics dock can export per-timepoint numbers.
        """
        frame_context = self._frame_context()
        self.histogram_widget.set_frame_source(frame_context, datasets)
        return slice_datasets(frame_context, datasets)

    def refresh_for_frame_change(self):
        """Re-feed the histogram after the displayed frame changed."""
        if not self.fret_layers:
            return
        self._update_fret_histogram(update_bounds=False)

    def _apply_fret_range_to_layers(self, layers, min_val, max_val):
        """Clip FRET output layers from their original data."""
        self._updating_linked_layers = True
        try:
            for layer in layers:
                if 'fret_data_original' not in layer.metadata:
                    continue
                original = layer.metadata['fret_data_original']
                layer.data = np.clip(original, min_val, max_val)
                layer.contrast_limits = [min_val, max_val]
            self.colormap_contrast_limits = [min_val, max_val]
        finally:
            self._updating_linked_layers = False

    def _on_fret_range_changed(self, min_val, max_val):
        """Handle range slider changes – clip FRET layers and update histogram."""
        selected_layers = list(
            self._fret_output_layers(selected_only=True).values()
        )
        self._apply_fret_range_to_layers(selected_layers, min_val, max_val)

        self.histogram_widget.update_colormap(
            colormap_colors=self.fret_colormap,
            contrast_limits=[min_val, max_val],
            gamma=self.colormap_gamma,
        )
        self._update_fret_histogram(update_bounds=False)

        self.plot_donor_trajectory()

    def _fret_filter_params(self, metric=None):
        """Return the donor trajectory a new efficiency criterion should use.

        The parameters are frozen into the criterion so that reloading a
        saved layer reproduces the same efficiencies, and refreshed from the
        tab on every recalculation so that a filter never keeps hiding pixels
        by a trajectory the user has already moved on from.
        """
        params = {}
        frequency = self._positive_float(self.frequency_input.text())
        donor_lifetime = self._positive_float(self.donor_line_edit.text())
        if frequency is None or donor_lifetime is None:
            return params
        params['frequency'] = frequency
        params['donor_lifetime'] = donor_lifetime
        params['donor_background'] = self.donor_background
        params['background_real'] = self.background_real
        params['background_imag'] = self.background_imag
        params['donor_fretting'] = self.donor_fretting_proportion
        return params

    @staticmethod
    def _positive_float(text):
        """Return *text* as a finite positive float, or ``None``."""
        try:
            value = float(str(text).strip())
        except (TypeError, ValueError):
            return None
        if not np.isfinite(value) or value <= 0:
            return None
        return value

    def _current_harmonic(self):
        """Return the harmonic a new criterion should be measured on."""
        return getattr(self.parent_widget, 'harmonic', 1) or 1

    def _filter_enable_blocked_reason(self):
        """Return why the efficiency filter cannot be switched on, else ``None``."""
        if not self._filter_layers():
            return "Select at least one image layer with phasor features."
        if not self._fret_filter_params():
            return (
                "Enter the donor lifetime and the frequency (MHz) before "
                "filtering on FRET efficiency."
            )
        return None

    def _refresh_filter_enable_state(self):
        """Explain on the card itself when the filter cannot be switched on."""
        self.filter_list.set_enable_blocked(
            self._filter_enable_blocked_reason()
        )

    def _filter_layers(self):
        """Return the layers the filter stack is written to."""
        if self.parent_widget is None:
            return []
        try:
            return list(self.parent_widget.get_selected_layers())
        except (AttributeError, RuntimeError):
            return []

    def _primary_filter_layer(self):
        """Return the layer whose stack the cards show, or ``None``."""
        layers = self._filter_layers()
        return layers[0] if layers else None

    def _layer_filter_params(self, layer):
        """Return *layer*'s own intensity filter/threshold parameters."""
        if self.parent_widget is None:
            return {}
        return self.parent_widget._filter_params_from_settings(layer)

    def _sync_filter_ui(self):
        """Show the stack stored on the primary layer."""
        layer = self._primary_filter_layer()
        if layer is None:
            self.filter_list.set_filters([])
            self.filter_list.set_filter_stats({}, "")
            self._refresh_filter_enable_state()
            return
        self.filter_list.set_filters(self._own_filters(get_filters(layer)))
        self._refresh_filter_stats()
        self._refresh_filter_enable_state()

    @staticmethod
    def _own_filters(filters):
        """Return the efficiency criteria of *filters*; this tab edits no others."""
        return [f for f in filters if f['metric'] == FRET_EFFICIENCY]

    def _refresh_filter_stats(self):
        """Report what the efficiency criterion keeps."""
        filters = self.filter_list.filters()
        layer = self._primary_filter_layer()
        if layer is None or not filters:
            self.filter_list.set_filter_stats({}, "")
            return
        mean, real, imag = baseline_arrays(
            layer, self._layer_filter_params(layer)
        )
        harmonics = layer.metadata.get('harmonics')
        stats = {}
        for entry in filters:
            single = dict(entry, enabled=True)
            mask = combined_mask([single], mean, real, imag, harmonics)
            prefix = "" if entry['enabled'] else "off · "
            stats[entry['id']] = (
                f"{prefix}keeps {kept_fraction(mask, mean):.1%} of the pixels"
            )
        self.filter_list.set_filter_stats(stats)

    def _refresh_fret_filter_params(self, layers=None):
        """Point every efficiency criterion at the current donor trajectory.

        The efficiency map on screen is rebuilt from the tab's live
        parameters; a criterion left on an older trajectory would hide a
        different set of pixels than the map it is displayed next to.
        """
        params = self._fret_filter_params()
        if not params:
            return False
        changed = False
        for layer in layers if layers is not None else self._filter_layers():
            filters = get_filters(layer)
            updated = []
            for entry in filters:
                if entry['metric'] == FRET_EFFICIENCY and (
                    entry['params'] != params
                ):
                    entry = dict(entry, params=dict(params))
                    changed = True
                updated.append(entry)
            if changed:
                set_filters(layer, updated)
        return changed

    def _on_filters_changed(self, filters):
        """Persist the edited stack and rebuild everything downstream of it."""
        self._apply_filter_stack(filters)

    def _apply_filter_stack(self, filters=None, layers=None):
        """Write *filters* to *layers* and re-derive their phasor data."""
        if self.parent_widget is None:
            return
        layers = self._filter_layers() if layers is None else list(layers)
        if not layers:
            return
        if filters is None:
            filters = self.filter_list.filters()
        own = self._own_filters(normalize_filters(filters))
        params = self._fret_filter_params()
        # A criterion first switched on from the placeholder card carries no
        # trajectory yet; it takes the tab's current one.
        own = [
            f if f['params'] or not params else dict(f, params=dict(params))
            for f in own
        ]

        problems = []
        self._applying_mapping_filter = True
        try:
            for layer in layers:
                # The Phasor Mapping tab's criteria are not shown here, but
                # they are part of the same stack and must survive untouched.
                others = [
                    f
                    for f in get_filters(layer)
                    if f['metric'] != FRET_EFFICIENCY
                ]
                stored = set_filters(layer, others + own)
                rebuild_layer_from_filters(
                    layer,
                    stored,
                    filter_params=self._layer_filter_params(layer),
                    on_error=problems.append,
                )
            self.parent_widget.refresh_phasor_data()
            if self.fret_layers:
                self.calculate_fret_efficiency()
        finally:
            self._applying_mapping_filter = False

        self._sync_filter_ui()
        for message in dict.fromkeys(problems):
            show_warning(message)

    def _fret_validation(self):
        """Return ``None`` if FRET efficiency can run, else the missing msg."""
        if not self.parent_widget.get_selected_layers():
            return "Select at least one image layer with phasor features."
        donor = self.donor_line_edit.text().strip()
        if not donor:
            return "Enter a donor lifetime value (or select donor layer(s))."
        frequency = self.frequency_input.text().strip()
        if not frequency:
            return "Enter the frequency (MHz)."
        try:
            float(donor)
            float(frequency)
        except ValueError:
            return "Donor lifetime and frequency must be valid numbers."
        return None

    def _on_image_layer_changed(self):
        """Callback whenever the image layer with phasor features changes."""
        self._teardown_on_layer_change()
        self._restore_on_layer_change()
        if hasattr(self, '_refresh_calculate_button'):
            self._refresh_calculate_button()

    def _teardown_on_layer_change(self):
        """Immediate cleanup: remove artists and disconnect signals."""
        if self.current_donor_line is not None:
            with contextlib.suppress(ValueError):
                self.current_donor_line.remove()
            self.current_donor_line = None

        if self.current_donor_circle is not None:
            with contextlib.suppress(ValueError):
                self.current_donor_circle.remove()
            self.current_donor_circle = None

        if self.current_background_circle is not None:
            with contextlib.suppress(ValueError):
                self.current_background_circle.remove()
            self.current_background_circle = None

        # Disconnect events from all FRET layers
        for layer in self.fret_layers:
            if layer in self.viewer.layers:
                try:
                    layer.events.colormap.disconnect(self._on_colormap_changed)
                    layer.events.contrast_limits.disconnect(
                        self._on_contrast_limits_changed
                    )
                    layer.events.gamma.disconnect(self._on_colormap_changed)
                except Exception:  # noqa: BLE001
                    pass

        self.fret_layer = None
        self.fret_layers = []
        self.fret_colormap = None
        self.colormap_contrast_limits = None

        self.histogram_widget.clear()

    def _restore_on_layer_change(self):
        """Deferred restore: update UI state from metadata."""
        self._needs_update = False

        layer_name = self.parent_widget.get_primary_layer_name()

        if layer_name:
            self._reconnect_existing_fret_layer(layer_name)
            self._sync_filter_ui()

            self._updating_settings = True
            try:
                self._restore_fret_settings_from_metadata()
                # Only plot the donor trajectory (visual element) but do NOT
                # run calculate_fret_efficiency. The user must click the
                # calculate button to run the analysis.
                if self.donor_line_edit.text() and self.frequency_input.text():
                    self.plot_donor_trajectory()
            finally:
                self._updating_settings = False

            self._previous_layer_name = layer_name
        else:
            self._updating_settings = True
            try:
                self.donor_line_edit.clear()
                self.frequency_input.clear()
                self.background_slider.setValue(10)
                self.background_label.setText("0.10")
                self.background_real_edit.setText("0.0")
                self.background_imag_edit.setText("0.0")
                self.fretting_slider.setValue(100)
                self.fretting_label.setText("1.00")
                self.colormap_checkbox.setChecked(True)

                self.background_positions_by_harmonic = {}
            finally:
                self._updating_settings = False

            self._previous_layer_name = None
            self._sync_filter_ui()

        self._refresh_settings_note()

    def calculate_fret_efficiency(self):
        """Calculate FRET efficiency based on donor intensities."""
        if not self.donor_line_edit.text().strip():
            show_warning("Enter a Donor lifetime value.")
            return

        if not self.frequency_input.text().strip():
            show_warning("Enter a frequency value.")
            return
        try:
            float(self.donor_line_edit.text().strip())
            float(self.frequency_input.text().strip())
        except ValueError:
            show_error(
                "Enter valid numeric values for donor lifetime and frequency."
            )
            return

        if not self.parent_widget.has_phasor_data():
            return

        selected_layers = self.parent_widget.get_selected_layers()
        if not selected_layers:
            return

        # An efficiency filter is only honest while it names the same donor
        # trajectory the map beside it was computed from, so the stack is
        # re-pointed at the current parameters before anything is computed.
        # (Skipped mid-apply: the stack is what asked for this run.)
        if not self._applying_mapping_filter and (
            self._refresh_fret_filter_params(selected_layers)
        ):
            self._apply_filter_stack(
                get_filters(selected_layers[0]), layers=selected_layers
            )
            return

        # Clear the active registry and disconnect its events.
        existing_outputs = self._fret_output_layers()
        self._set_fret_layers([])

        # Save settings to metadata for primary layer
        primary_layer_name = self.parent_widget.get_primary_layer_name()
        if (
            not hasattr(self, '_updating_settings')
            or not self._updating_settings
        ):
            try:
                if self.donor_line_edit.text():
                    donor_lifetime = float(self.donor_line_edit.text())
                    self.donor_lifetime = donor_lifetime
                    self._update_fret_setting_in_metadata(
                        'donor_lifetime', donor_lifetime
                    )
            except ValueError:
                pass

            if primary_layer_name:
                self._update_fret_setting_in_metadata(
                    'donor_background', self.donor_background
                )
                self._update_fret_setting_in_metadata(
                    'donor_fretting_proportion', self.donor_fretting_proportion
                )
                self._update_fret_setting_in_metadata(
                    'use_colormap', self.use_colormap
                )
                self._update_fret_setting_in_metadata(
                    'background_positions_by_harmonic',
                    self.background_positions_by_harmonic,
                )

        if not hasattr(self, 'donor_lifetime') or self.donor_lifetime is None:
            try:
                self.donor_lifetime = float(self.donor_line_edit.text())
            except ValueError:
                show_error("Please enter a valid donor lifetime")
                return

        # Get or calculate donor trajectory
        if hasattr(self.current_donor_line, 'get_xydata'):
            donor_trajectory = self.current_donor_line.get_xydata()
            neighbor_real = donor_trajectory[:, 0]
            neighbor_imag = donor_trajectory[:, 1]
        else:
            # Calculate donor trajectory if not available
            base_frequency = float(self.frequency_input.text().strip())
            effective_frequency = base_frequency * self.parent_widget.harmonic
            neighbor_real, neighbor_imag = phasor_from_fret_donor(
                frequency=effective_frequency,
                donor_lifetime=self.donor_lifetime,
                fret_efficiency=self._fret_efficiencies,
                donor_background=self.donor_background,
                background_real=self.background_real,
                background_imag=self.background_imag,
                donor_fretting=self.donor_fretting_proportion,
            )

        # The nearest-neighbour search is the expensive step here and is
        # independent per layer, so every efficiency map is computed up front
        # in a thread pool. The harmonic is read once, on this thread,
        # because it comes from a Qt spinbox.
        harmonic = self.parent_widget.harmonic

        # Each layer is compared with the donor trajectory at the frequency
        # it was acquired with: the primary at the entered one, the others
        # at their stored one.
        entered_frequency = float(self.frequency_input.text().strip())
        trajectories = {}
        layer_trajectories = {}
        for layer in selected_layers:
            frequency = (
                self.parent_widget.layer_frequency(layer, entered_frequency)
                if hasattr(self.parent_widget, 'layer_frequency')
                else entered_frequency
            )
            if np.isclose(frequency, entered_frequency):
                layer_trajectories[layer.name] = (neighbor_real, neighbor_imag)
                continue
            if frequency not in trajectories:
                trajectories[frequency] = phasor_from_fret_donor(
                    frequency=frequency * harmonic,
                    donor_lifetime=self.donor_lifetime,
                    fret_efficiency=self._fret_efficiencies,
                    donor_background=self.donor_background,
                    background_real=self.background_real,
                    background_imag=self.background_imag,
                    donor_fretting=self.donor_fretting_proportion,
                )
            layer_trajectories[layer.name] = trajectories[frequency]

        def compute_efficiency(layer):
            """Return one layer's FRET efficiency map, or ``None``.

            Pure array work, safe to run in a worker thread.
            """
            trajectory_real, trajectory_imag = layer_trajectories[layer.name]
            g_array = layer.metadata.get("G")
            s_array = layer.metadata.get("S")
            harmonics = layer.metadata.get("harmonics")

            if g_array is None or s_array is None:
                return None

            if harmonics is not None and g_array.ndim > layer.data.ndim:
                try:
                    harmonics_array = np.atleast_1d(harmonics)
                    harmonic_index = np.where(harmonics_array == harmonic)[0][
                        0
                    ]
                    real = g_array[harmonic_index]
                    imag = s_array[harmonic_index]
                except IndexError:
                    return None
            else:
                real = g_array
                imag = s_array

            return phasor_nearest_neighbor(
                real,
                imag,
                trajectory_real,
                trajectory_imag,
                values=self._fret_efficiencies,
            )

        efficiencies = parallel_map(
            compute_efficiency, selected_layers, on_error="collect"
        )

        if not self._updating_settings and self._has_settings_store():
            # The run's parameters become every analysed layer's settings.
            analysed_layers = [
                layer
                for layer, efficiency in zip(
                    selected_layers, efficiencies, strict=True
                )
                if efficiency is not None
                and not isinstance(efficiency, BaseException)
            ]
            self._commit_fret_settings(analysed_layers)
            self.parent_widget.commit_frequency(
                analysed_layers, entered_frequency
            )

        # Process each selected layer
        for layer, fret_efficiency in zip(
            selected_layers, efficiencies, strict=True
        ):
            if isinstance(fret_efficiency, BaseException):
                show_error(
                    f"FRET efficiency failed for {layer.name}: "
                    f"{fret_efficiency}"
                )
                continue
            if fret_efficiency is None:
                continue

            fret_layer_name = f"FRET efficiency: {layer.name}"

            fret_layer = existing_outputs.get(layer.name)

            # The saved display (kept in step with the user's changes by
            # ``_remember_fret_display``) wins; without one, a layer that is
            # already shown keeps its own, and only a new one gets defaults.
            display_colormap = 'viridis'
            display_contrast_limits = (0, 1)
            display_gamma = None
            if (
                hasattr(self, '_saved_colormap_name')
                and not self._updating_settings
            ):
                display_colormap = self._saved_fret_colormap()
                display_contrast_limits = tuple(self._saved_contrast_limits)
                display_gamma = getattr(self, '_saved_gamma', None)
            elif fret_layer is not None:
                display_colormap = fret_layer.colormap
                display_contrast_limits = tuple(fret_layer.contrast_limits)
                display_gamma = fret_layer.gamma

            if fret_layer is None:
                selected_fret_layer = Image(
                    fret_efficiency,
                    name=fret_layer_name,
                    scale=layer.scale,
                    colormap=display_colormap,
                    contrast_limits=display_contrast_limits,
                )
                fret_layer = self.viewer.add_layer(selected_fret_layer)
            else:
                fret_layer.data = fret_efficiency
                fret_layer.scale = layer.scale
                fret_layer.colormap = display_colormap
                fret_layer.contrast_limits = display_contrast_limits
            if display_gamma is not None:
                fret_layer.gamma = display_gamma

            fret_layer.metadata['fret_data_original'] = fret_efficiency.copy()
            fret_layer.metadata[_FRET_OUTPUT_METADATA_KEY] = {
                'source_layer': layer.name
            }

            # Add to list of FRET layers and connect events
            self.fret_layers.append(fret_layer)
            fret_layer.events.colormap.connect(self._on_colormap_changed)
            fret_layer.events.contrast_limits.connect(
                self._on_contrast_limits_changed
            )
            fret_layer.events.gamma.connect(self._on_colormap_changed)

            # Store reference to first FRET layer for backward compatibility
            if self.fret_layer is None:
                self.fret_layer = fret_layer
                self.fret_colormap = fret_layer.colormap.colors
                self.colormap_contrast_limits = fret_layer.contrast_limits
                self.colormap_gamma = fret_layer.gamma

        self._set_fret_layers(self._fret_output_layers().values())

        if (
            not hasattr(self, '_saved_colormap_name')
            or self._updating_settings
        ) and self.fret_layer is not None:
            self._update_fret_setting_in_metadata(
                'colormap_settings.colormap_name',
                self.fret_layer.colormap.name,
            )
            # A built-in colormap is restored by name; only a custom one
            # needs its colours to come back in another session.
            colormap = self.fret_layer.colormap
            colors = np.asarray(colormap.colors).tolist()
            is_builtin = (
                layer_colormap_from_settings(
                    {'colormap_name': colormap.name, 'colormap_colors': colors}
                )
                == colormap.name
            )
            self._update_fret_setting_in_metadata(
                'colormap_settings.colormap_colors',
                None if is_builtin else colors,
            )
            self._update_fret_setting_in_metadata(
                'colormap_settings.contrast_limits',
                [float(v) for v in self.fret_layer.contrast_limits],
            )
            self._update_fret_setting_in_metadata(
                'colormap_settings.gamma', self.fret_layer.gamma
            )
            self._update_fret_setting_in_metadata(
                'colormap_settings.colormap_changed', False
            )

        self._update_fret_histogram()
        self.plot_donor_trajectory()
        self._sync_filter_ui()

    def closeEvent(self, event):
        """Clean up signal connections before closing."""
        self._set_fret_layers([])

        # Disconnect viewer events
        with contextlib.suppress(TypeError, ValueError, AttributeError):
            self.viewer.layers.events.inserted.disconnect(
                self._on_layer_changed
            )
        with contextlib.suppress(TypeError, ValueError, AttributeError):
            self.viewer.layers.events.removed.disconnect(
                self._on_layer_changed
            )

        for layer in self._name_event_layers.values():
            with contextlib.suppress(TypeError, ValueError, AttributeError):
                layer.events.name.disconnect(self._update_background_combobox)
            with contextlib.suppress(TypeError, ValueError, AttributeError):
                layer.events.name.disconnect(
                    self._update_donor_lifetime_combobox
                )
        self._name_event_layers.clear()

        event.accept()


def draw_fret_trajectory_overlay(
    ax, trajectory_real, trajectory_imag, fret_efficiencies, settings=None
):
    """Stateless function to draw a FRET trajectory on a matplotlib axes."""

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap

    if settings is None:
        settings = {}

    use_colormap = settings.get("use_colormap", True)
    fret_colormap = settings.get("fret_colormap")
    colormap_contrast_limits = settings.get("colormap_contrast_limits", (0, 1))
    trajectory_zorder = settings.get("trajectory_zorder", 10)
    dot_zorder = settings.get("dot_zorder", 11)
    circle_radius = settings.get("circle_radius", 0.02)
    colormap_density_factor = settings.get("colormap_density_factor", 1.0)

    if use_colormap:
        if fret_colormap is not None:
            if isinstance(fret_colormap, list) and len(fret_colormap) <= 32:
                colormap = LinearSegmentedColormap.from_list(
                    "fret_interp", fret_colormap, N=256
                )
            elif isinstance(fret_colormap, list):
                colormap = ListedColormap(fret_colormap)
            else:
                colormap = plt.cm.jet
        else:
            colormap = plt.cm.jet

        donor_color = colormap(0.0)[:3]
        background_color = colormap(1.0)[:3]

        num_segments = min(
            int(len(trajectory_real) * colormap_density_factor),
            len(trajectory_real) - 1,
        )
        if num_segments < 1:
            num_segments = 1

        vmin, vmax = colormap_contrast_limits

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

            fret_value = fret_efficiencies[start_idx]
            colors.append(fret_value)

        lc = LineCollection(
            segments, cmap=colormap, linewidths=3, zorder=trajectory_zorder
        )
        lc.set_array(np.array(colors))
        lc.set_clim(vmin, vmax)
        ax.add_collection(lc)
    else:
        donor_color = 'dimgray'
        background_color = 'dimgray'
        ax.plot(
            trajectory_real,
            trajectory_imag,
            color='dimgray',
            linewidth=3,
            label='Donor Trajectory',
            zorder=trajectory_zorder,
        )

    donor_circle = plt.Circle(
        (trajectory_real[0], trajectory_imag[0]),
        circle_radius,
        fill=True,
        facecolor=donor_color,
        linewidth=1,
        zorder=dot_zorder,
    )
    ax.add_patch(donor_circle)

    background_circle = plt.Circle(
        (trajectory_real[-1], trajectory_imag[-1]),
        circle_radius,
        fill=True,
        facecolor=background_color,
        linewidth=1,
        zorder=dot_zorder,
    )
    ax.add_patch(background_circle)
