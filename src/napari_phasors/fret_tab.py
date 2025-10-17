import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from napari.layers import Image
from napari.utils.notifications import show_error, show_warning
from phasorpy.lifetime import (
    phasor_from_fret_donor,
    phasor_to_apparent_lifetime,
    phasor_to_normal_lifetime,
)
from phasorpy.phasor import phasor_center, phasor_nearest_neighbor
from qtpy.QtCore import Qt
from qtpy.QtGui import QDoubleValidator
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ._utils import update_frequency_in_metadata


class FretWidget(QWidget):
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
        self.fret_layer = None
        self.colormap_contrast_limits = None
        self.fret_colormap = None
        self.use_colormap = True
        self.colormap_density_factor = (
            5  # Controls trajectory colormap detail level
        )
        self.current_donor_circle = None
        self.current_background_circle = None
        self._updating_settings = False

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

        # Use a compact form layout for essentials
        form = QFormLayout()
        # Let fields grow by default but still shrink when needed
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

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
        self.donor_source_selector.addItems(["Manual", "From layer"])
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
        self.donor_stack = QStackedWidget()

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

        self.donor_lifetime_combobox = QComboBox()
        self.donor_lifetime_combobox.setMinimumContentsLength(8)
        self.donor_lifetime_combobox.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon
        )
        self.donor_lifetime_combobox.currentIndexChanged.connect(
            self._calculate_donor_lifetime
        )
        self.donor_lifetime_combobox.setToolTip(
            "Select the layer from which to derive the donor lifetime"
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

        # Background position source selector
        bg_source_row = QHBoxLayout()
        self.bg_source_selector = QComboBox()
        self.bg_source_selector.addItems(["Manual", "From layer"])
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
        self.bg_stack = QStackedWidget()

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

        self.background_image_combobox = QComboBox()
        self.background_image_combobox.setMinimumContentsLength(8)
        self.background_image_combobox.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon
        )
        self.background_image_combobox.currentIndexChanged.connect(
            self._calculate_background_position
        )
        self.background_image_combobox.setToolTip(
            "Select the layer from which to derive the background position"
        )
        bg_image_layout.addWidget(self.background_image_combobox)
        self.bg_stack.addWidget(bg_image_page)

        # Dynamic background label that changes based on mode
        self.background_position_label = QLabel("Background position:")
        form.addRow(self.background_position_label, self.bg_stack)

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

        # Add form to root layout
        layout.addLayout(form)

        # Colormap over trajectory checkbox
        self.colormap_checkbox = QCheckBox(
            "Overlay colormap on donor trajectory"
        )
        self.colormap_checkbox.setChecked(True)
        self.colormap_checkbox.stateChanged.connect(
            self._on_colormap_checkbox_changed
        )
        layout.addWidget(self.colormap_checkbox)

        # Plot button
        self.calculate_fret_efficiency_button = QPushButton(
            "Calculate FRET efficiency"
        )
        self.calculate_fret_efficiency_button.clicked.connect(
            self.calculate_fret_efficiency
        )
        layout.addWidget(self.calculate_fret_efficiency_button)

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

        # Update the label text based on the mode
        if index == 0:  # Manual
            self.donor_label.setText("Donor lifetime (ns):")
        else:  # From layer
            self.donor_label.setText("Donor lifetime (ns):")

        # When switching to layer-based mode, try computing from selection
        if index == 1:
            self._calculate_donor_lifetime()

    def _on_bg_source_changed(self, index: int):
        """Switch background position input mode (Manual | From image)."""
        if hasattr(self, 'bg_stack'):
            self.bg_stack.setCurrentIndex(index)

        # Update the label text based on the mode
        if index == 0:  # Manual
            self.background_position_label.setText("Background position:")
        else:  # From layer
            self.background_position_label.setText("Background position:")

        # When switching to image-based mode, try computing from selection
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

    def _on_harmonic_changed(self):
        """Handle harmonic changes from the parent widget."""
        if self.parent_widget is None:
            return

        new_harmonic = self.parent_widget.harmonic

        if self.current_harmonic != new_harmonic:
            was_updating = getattr(self, '_updating_settings', False)
            self._updating_settings = True
            try:
                self._store_current_background_position()
            finally:
                self._updating_settings = was_updating

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
                self.background_positions_by_harmonic[
                    self.current_harmonic
                ] = {'real': real, 'imag': imag}
                
                layer_name = self.parent_widget.image_layer_with_phasor_features_combobox.currentText()
                if layer_name and (not hasattr(self, '_updating_settings') or not self._updating_settings):
                    current_layer = self.viewer.layers[layer_name]
                    if 'settings' not in current_layer.metadata:
                        current_layer.metadata['settings'] = {}
                    if 'fret' not in current_layer.metadata['settings']:
                        current_layer.metadata['settings']['fret'] = self._get_default_fret_settings()
                    
                    current_layer.metadata['settings']['fret']['background_positions_by_harmonic'] = self.background_positions_by_harmonic.copy()
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
        try:
            real = float(self.background_real_edit.text().strip())
            imag = float(self.background_imag_edit.text().strip())
            self.background_real = real
            self.background_imag = imag
            
            self.background_positions_by_harmonic[self.current_harmonic] = {
                'real': real,
                'imag': imag
            }
            
            if (not hasattr(self, '_updating_settings') or not self._updating_settings):
                layer_name = self.parent_widget.image_layer_with_phasor_features_combobox.currentText()
                if layer_name:
                    self._update_fret_setting_in_metadata('background_real', real)
                    self._update_fret_setting_in_metadata('background_imag', imag)
                    self._update_fret_setting_in_metadata('background_positions_by_harmonic', self.background_positions_by_harmonic)
        except ValueError:
            pass
        
        self._on_parameters_changed()

    def _on_background_slider_changed(self):
        """Handle background slider value change."""
        value = self.background_slider.value() / 100.0
        self.donor_background = value
        self.background_label.setText(f"{value:.2f}")
        
        if not hasattr(self, '_updating_settings') or not self._updating_settings:
            self._update_fret_setting_in_metadata('donor_background', value)
        
        self._on_parameters_changed()

    def _on_fretting_slider_changed(self):
        """Handle fretting proportion slider value change."""
        value = self.fretting_slider.value() / 100.0
        self.donor_fretting_proportion = value
        self.fretting_label.setText(f"{value:.2f}")
        
        if not hasattr(self, '_updating_settings') or not self._updating_settings:
            self._update_fret_setting_in_metadata('donor_fretting_proportion', value)
        
        self._on_parameters_changed()

    def _on_colormap_checkbox_changed(self):
        """Handle colormap checkbox state change."""
        self.use_colormap = self.colormap_checkbox.isChecked()
        
        if not hasattr(self, '_updating_settings') or not self._updating_settings:
            self._update_fret_setting_in_metadata('use_colormap', self.use_colormap)
        
        self.plot_donor_trajectory()

    def _on_parameters_changed(self):
        """Update plot when any parameter changes."""
        # Only store values if we have a valid layer selected
        layer_name = self.parent_widget.image_layer_with_phasor_features_combobox.currentText()
        if not layer_name:
            return
            
        if self.donor_line_edit.text():
            try:
                self.donor_lifetime = float(self.donor_line_edit.text())
                if not hasattr(self, '_updating_settings') or not self._updating_settings:
                    self._update_fret_setting_in_metadata('donor_lifetime', self.donor_lifetime)
            except ValueError:
                pass
        
        if self.frequency_input.text():
            try:
                base_frequency = float(self.frequency_input.text().strip())
                if not hasattr(self, '_updating_settings') or not self._updating_settings:
                    self._update_fret_setting_in_metadata('frequency', base_frequency)
            except ValueError:
                pass
        
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
        current_text = self.background_image_combobox.currentText()
        self.background_image_combobox.blockSignals(True)
        self.background_image_combobox.clear()

        self.background_image_combobox.addItem("Select layer...")

        from napari.layers import Image

        layer_names = [
            layer.name
            for layer in self.viewer.layers
            if isinstance(layer, Image)
            and "phasor_features_labels_layer" in layer.metadata.keys()
        ]

        self.background_image_combobox.addItems(layer_names)

        if current_text in layer_names:
            index = self.background_image_combobox.findText(current_text)
            if index >= 0:
                self.background_image_combobox.setCurrentIndex(index)
        elif current_text == "Select layer..." or current_text == "":
            self.background_image_combobox.setCurrentIndex(0)

        self.background_image_combobox.blockSignals(False)

    def _on_layer_changed(self):
        """Handle when layers are added or removed in the viewer."""
        self._update_background_combobox()
        self._update_donor_lifetime_combobox()

    def _calculate_background_position(self):
        """Calculate the background position from selected background image layer for all harmonics."""
        background_layer_name = self.background_image_combobox.currentText()
        if (
            not background_layer_name
            or background_layer_name == "Select layer..."
        ):
            if self.bg_source_selector.currentIndex() == 1:
                self.background_position_label.setText("Background position:")
            return

        try:
            background_layer = self.viewer.layers[background_layer_name]
            background_phasor_layer = background_layer.metadata[
                "phasor_features_labels_layer"
            ]
        except (KeyError, AttributeError):
            if self.bg_source_selector.currentIndex() == 1:
                self.background_position_label.setText("Background position:")
            return

        phasor_data = background_phasor_layer.features

        if 'harmonic' not in phasor_data.columns:
            if self.bg_source_selector.currentIndex() == 1:
                self.background_position_label.setText("Background position:")
            return

        unique_harmonics = phasor_data['harmonic'].unique()

        for harmonic in unique_harmonics:
            harmonic_mask = phasor_data['harmonic'] == harmonic
            real = phasor_data.loc[harmonic_mask, 'G']
            imag = phasor_data.loc[harmonic_mask, 'S']

            if len(real) == 0 or len(imag) == 0:
                continue

            try:
                _, center_real, center_imag = phasor_center(
                    background_layer.data.flatten(), real, imag
                )

                self.background_positions_by_harmonic[harmonic] = {
                    'real': center_real,
                    'imag': center_imag,
                }
            except Exception:
                continue

        current_harmonic = self.parent_widget.harmonic
        if current_harmonic in self.background_positions_by_harmonic:
            stored = self.background_positions_by_harmonic[current_harmonic]
            self.background_real_edit.setText(f"{stored['real']:.3f}")
            self.background_imag_edit.setText(f"{stored['imag']:.3f}")
            self.background_real = stored['real']
            self.background_imag = stored['imag']

            # Update the label to show calculated values
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

    def _update_donor_lifetime_combobox(self):
        """Update the donor lifetime combobox with available layers."""
        current_text = self.donor_lifetime_combobox.currentText()
        self.donor_lifetime_combobox.blockSignals(True)
        self.donor_lifetime_combobox.clear()

        self.donor_lifetime_combobox.addItem("Select layer...")

        layer_names = [
            layer.name
            for layer in self.viewer.layers
            if isinstance(layer, Image)
            and "phasor_features_labels_layer" in layer.metadata.keys()
        ]

        self.donor_lifetime_combobox.addItems(layer_names)

        if current_text in layer_names:
            index = self.donor_lifetime_combobox.findText(current_text)
            if index >= 0:
                self.donor_lifetime_combobox.setCurrentIndex(index)
        elif current_text == "Select layer..." or current_text == "":
            self.donor_lifetime_combobox.setCurrentIndex(0)

        self.donor_lifetime_combobox.blockSignals(False)

    def _calculate_donor_lifetime(self):
        """Calculate the donor lifetime from selected layer for current harmonic."""
        donor_layer_name = self.donor_lifetime_combobox.currentText()
        if not donor_layer_name or donor_layer_name == "Select layer...":
            if self.donor_source_selector.currentIndex() == 1:
                self.donor_label.setText("Donor lifetime (ns):")
            return

        try:
            donor_layer = self.viewer.layers[donor_layer_name]
            donor_phasor_layer = donor_layer.metadata[
                "phasor_features_labels_layer"
            ]
        except (KeyError, AttributeError):
            if self.donor_source_selector.currentIndex() == 1:
                self.donor_label.setText("Donor lifetime (ns):")
            return

        phasor_data = donor_phasor_layer.features

        if "harmonic" not in phasor_data.columns:
            if self.donor_source_selector.currentIndex() == 1:
                self.donor_label.setText("Donor lifetime (ns):")
            return

        current_harmonic = self.parent_widget.harmonic
        harmonic_mask = phasor_data['harmonic'] == current_harmonic
        real = phasor_data.loc[harmonic_mask, 'G']
        imag = phasor_data.loc[harmonic_mask, 'S']

        if len(real) == 0 or len(imag) == 0:
            if self.donor_source_selector.currentIndex() == 1:
                self.donor_label.setText("Donor lifetime (ns):")
            return

        try:
            _, center_real, center_imag = phasor_center(
                donor_layer.data.flatten(), real, imag
            )

            if not self.frequency_input.text():
                if self.donor_source_selector.currentIndex() == 1:
                    self.donor_label.setText("Donor lifetime (ns):")
                return

            frequency = float(self.frequency_input.text().strip())

            lifetime_type = self.lifetime_type_combobox.currentText()

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
                if self.donor_source_selector.currentIndex() == 1:
                    self.donor_label.setText("Donor lifetime (ns):")
                return

            self.donor_line_edit.setText(f"{lifetime:.2f}")
            self.donor_lifetime = lifetime

            # Update the label to show calculated value
            if self.donor_source_selector.currentIndex() == 1:
                self.donor_label.setText(
                    f"Donor lifetime (from layer): {lifetime:.2f} ns"
                )

            self.plot_donor_trajectory()

        except Exception as e:
            show_error(f"Error calculating donor lifetime: {str(e)}")
            if self.donor_source_selector.currentIndex() == 1:
                self.donor_label.setText("Donor lifetime (ns): Error")

    def plot_donor_trajectory(self):
        """Plot the donor trajectory with current parameters."""
        try:
            if self.current_donor_line is not None:
                try:
                    self.current_donor_line.remove()
                except ValueError:
                    pass
                self.current_donor_line = None

            if self.current_donor_circle is not None:
                try:
                    self.current_donor_circle.remove()
                except ValueError:
                    pass
                self.current_donor_circle = None

            if self.current_background_circle is not None:
                try:
                    self.current_background_circle.remove()
                except ValueError:
                    pass
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
                    colormap = plt.cm.turbo

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

        except Exception as e:
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
            colormap = plt.cm.turbo

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
        lc.set_clim(vmin, vmax)

        self.current_donor_line = ax.add_collection(lc)

    def _on_colormap_changed(self, event):
        """Handle changes to the colormap of the FRET layer."""
        if self.fret_layer is not None:
            layer = event.source
            self.fret_colormap = layer.colormap.colors
            self.colormap_contrast_limits = layer.contrast_limits
            
            colormap_name = getattr(layer.colormap, 'name', 'custom')
            colormap_colors = getattr(layer.colormap, 'colors', None)
            
            if colormap_colors is not None:
                if hasattr(colormap_colors, 'tolist'):
                    colormap_colors = colormap_colors.tolist()
                elif isinstance(colormap_colors, np.ndarray):
                    colormap_colors = colormap_colors.tolist()
            
            self._update_fret_setting_in_metadata('colormap_settings.colormap_name', colormap_name)
            self._update_fret_setting_in_metadata('colormap_settings.colormap_colors', colormap_colors)
            self._update_fret_setting_in_metadata('colormap_settings.colormap_changed', True)

            self.plot_donor_trajectory()

    def _on_contrast_limits_changed(self, event):
        """Handle changes to the contrast limits of the FRET layer."""
        if self.fret_layer is not None:
            layer = event.source
            self.colormap_contrast_limits = layer.contrast_limits
            
            contrast_limits = layer.contrast_limits
            if hasattr(contrast_limits, 'tolist'):
                contrast_limits = contrast_limits.tolist()
            elif isinstance(contrast_limits, np.ndarray):
                contrast_limits = contrast_limits.tolist()
            
            self._update_fret_setting_in_metadata('colormap_settings.contrast_limits', contrast_limits)

            self.plot_donor_trajectory()

    def _get_default_fret_settings(self):
        """Get default settings dictionary for FRET parameters."""
        return {
            'donor_lifetime': None,
            'frequency': None,
            'donor_background': 0.1,
            'background_real': 0.0,
            'background_imag': 0.0,
            'donor_fretting_proportion': 1.0,
            'use_colormap': True,
            'analysis_performed': False,
            'background_positions_by_harmonic': {},
            'colormap_settings': {
                'colormap_name': 'viridis',
                'colormap_colors': None,
                'contrast_limits': (0, 1),
                'colormap_changed': False
            }
        }

    def _initialize_fret_settings_in_metadata(self, layer):
        """Initialize FRET settings in layer metadata if not present."""
        if 'settings' not in layer.metadata:
            layer.metadata['settings'] = {}
        
        if 'fret' not in layer.metadata['settings']:
            layer.metadata['settings']['fret'] = self._get_default_fret_settings()

    def _update_fret_setting_in_metadata(self, key_path, value):
        """Update a specific FRET setting in the current layer's metadata."""
        if hasattr(self, '_updating_settings') and self._updating_settings:
            return
            
        layer_name = self.parent_widget.image_layer_with_phasor_features_combobox.currentText()
        if not layer_name:
            return
            
        try:
            layer = self.viewer.layers[layer_name]
            
            if 'settings' not in layer.metadata:
                layer.metadata['settings'] = {}
            if 'fret' not in layer.metadata['settings']:
                layer.metadata['settings']['fret'] = self._get_default_fret_settings()
            
            keys = key_path.split('.')
            current_dict = layer.metadata['settings']['fret']
            
            for key in keys[:-1]:
                if key not in current_dict:
                    current_dict[key] = {}
                current_dict = current_dict[key]
            
            current_dict[keys[-1]] = value
        except (KeyError, AttributeError):
            pass

    def _restore_fret_settings_from_metadata(self):
        """Restore all FRET settings from the current layer's metadata."""
        layer_name = self.parent_widget.image_layer_with_phasor_features_combobox.currentText()
        if not layer_name:
            return
            
        layer = self.viewer.layers[layer_name]
        if 'settings' not in layer.metadata or 'fret' not in layer.metadata['settings']:
            self._initialize_fret_settings_in_metadata(layer)
            return
            
        self._updating_settings = True
        try:
            settings = layer.metadata['settings']['fret']
            
            self.donor_line_edit.clear()
            self.frequency_input.clear()
            
            if settings.get('donor_lifetime') is not None:
                self.donor_lifetime = settings['donor_lifetime']
                self.donor_line_edit.setText(str(self.donor_lifetime))
            
            if settings.get('frequency') is not None:
                base_frequency = settings['frequency']
                self.frequency_input.setText(str(base_frequency))
                self.frequency = base_frequency * self.parent_widget.harmonic
            
            if settings.get('donor_background') is not None:
                self.donor_background = settings['donor_background']
                self.background_slider.setValue(int(self.donor_background * 100))
                self.background_label.setText(f"{self.donor_background:.2f}")
            
            if settings.get('background_positions_by_harmonic'):
                self.background_positions_by_harmonic = settings['background_positions_by_harmonic'].copy()
            
            self._load_background_position_for_harmonic(self.current_harmonic)
            
            if settings.get('donor_fretting_proportion') is not None:
                self.donor_fretting_proportion = settings['donor_fretting_proportion']
                self.fretting_slider.setValue(int(self.donor_fretting_proportion * 100))
                self.fretting_label.setText(f"{self.donor_fretting_proportion:.2f}")
            
            if settings.get('use_colormap') is not None:
                self.use_colormap = settings['use_colormap']
                self.colormap_checkbox.setChecked(self.use_colormap)
            
            if 'colormap_settings' in settings:
                colormap_settings = settings['colormap_settings']
                self._saved_colormap_name = colormap_settings.get('colormap_name', 'viridis')
                self._saved_colormap_colors = colormap_settings.get('colormap_colors', None)
                self._saved_contrast_limits = colormap_settings.get('contrast_limits', (0, 1))
                self._colormap_was_changed = colormap_settings.get('colormap_changed', False)
                        
        finally:
            self._updating_settings = False

    def _recreate_fret_from_metadata(self):
        """Recreate FRET analysis from metadata if it was previously performed."""
        layer_name = self.parent_widget.image_layer_with_phasor_features_combobox.currentText()
        if layer_name:
            layer = self.viewer.layers[layer_name]
            if ('settings' in layer.metadata and 
                'fret' in layer.metadata['settings'] and
                layer.metadata['settings']['fret'].get('analysis_performed', False)):
                
                if (self.donor_line_edit.text() and self.frequency_input.text()):
                    self._updating_settings = True
                    try:
                        self.calculate_fret_efficiency()
                        if hasattr(self, '_saved_colormap_name'):
                            self._apply_saved_fret_colormap_settings()
                    finally:
                        self._updating_settings = False
            else:
                if (self.donor_line_edit.text() and self.frequency_input.text()):
                    self.plot_donor_trajectory()

    def _apply_saved_fret_colormap_settings(self):
        """Apply saved colormap settings to FRET layer if it exists."""
        if (self.fret_layer is not None and 
            hasattr(self, '_saved_colormap_name')):
            
            try:
                self.fret_layer.events.colormap.disconnect(self._on_colormap_changed)
                self.fret_layer.events.contrast_limits.disconnect(self._on_contrast_limits_changed)
                
                if self._saved_colormap_colors is not None:
                    from napari.utils.colormaps import Colormap
                    
                    if isinstance(self._saved_colormap_colors, list):
                        saved_colors = np.array(self._saved_colormap_colors)
                    else:
                        saved_colors = self._saved_colormap_colors
                    
                    saved_colormap = Colormap(colors=saved_colors, name="saved_custom")
                    self.fret_layer.colormap = saved_colormap
                else:
                    self.fret_layer.colormap = self._saved_colormap_name
                
                if isinstance(self._saved_contrast_limits, list):
                    saved_limits = tuple(self._saved_contrast_limits)
                else:
                    saved_limits = self._saved_contrast_limits
                
                self.fret_layer.contrast_limits = saved_limits
                
                self.fret_colormap = self.fret_layer.colormap.colors
                self.colormap_contrast_limits = self.fret_layer.contrast_limits
                
                self.fret_layer.events.colormap.connect(self._on_colormap_changed)
                self.fret_layer.events.contrast_limits.connect(self._on_contrast_limits_changed)
                
                self.plot_donor_trajectory()
                
            except Exception as e:
                print(f"Error applying saved colormap settings: {e}")
                try:
                    self.fret_layer.events.colormap.connect(self._on_colormap_changed)
                    self.fret_layer.events.contrast_limits.connect(self._on_contrast_limits_changed)
                except Exception:
                    pass

    def _reconnect_existing_fret_layer(self, layer_name):
        """Reconnect to existing FRET layer if it exists for this layer."""
        fret_layer_name = f"FRET efficiency: {layer_name}"
        
        if fret_layer_name in self.viewer.layers:
            self.fret_layer = self.viewer.layers[fret_layer_name]
            
            self.fret_layer.events.colormap.connect(self._on_colormap_changed)
            self.fret_layer.events.contrast_limits.connect(self._on_contrast_limits_changed)
            
            if hasattr(self, '_saved_colormap_name'):
                self._apply_saved_fret_colormap_settings()
            else:
                self.fret_colormap = self.fret_layer.colormap.colors
                self.colormap_contrast_limits = self.fret_layer.contrast_limits

    def _on_image_layer_changed(self):
        """Callback whenever the image layer with phasor features changes."""
        if self.current_donor_line is not None:
            try:
                self.current_donor_line.remove()
            except ValueError:
                pass
            self.current_donor_line = None

        if self.current_donor_circle is not None:
            try:
                self.current_donor_circle.remove()
            except ValueError:
                pass
            self.current_donor_circle = None

        if self.current_background_circle is not None:
            try:
                self.current_background_circle.remove()
            except ValueError:
                pass
            self.current_background_circle = None

        if self.fret_layer is not None:
            try:
                self.fret_layer.events.colormap.disconnect(self._on_colormap_changed)
                self.fret_layer.events.contrast_limits.disconnect(self._on_contrast_limits_changed)
            except Exception:
                pass
        
        self.fret_layer = None
        self.fret_colormap = None
        self.colormap_contrast_limits = None
        
        layer_name = self.parent_widget.image_layer_with_phasor_features_combobox.currentText()
        
        if layer_name:
            layer = self.viewer.layers[layer_name]
            self._initialize_fret_settings_in_metadata(layer)
            
            self._reconnect_existing_fret_layer(layer_name)
            
            self._updating_settings = True
            try:
                # Reset background positions before restoring from metadata
                self.background_positions_by_harmonic = {}
                self._restore_fret_settings_from_metadata()
                self._recreate_fret_from_metadata()
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
        if self.parent_widget._labels_layer_with_phasor_features is None:
            return
        labels_layer_name = (
            self.parent_widget.image_layer_with_phasor_features_combobox.currentText()
        )
        if not labels_layer_name:
            return

        if not hasattr(self, '_updating_settings') or not self._updating_settings:
            try:
                if self.donor_line_edit.text():
                    donor_lifetime = float(self.donor_line_edit.text())
                    self.donor_lifetime = donor_lifetime
                    self._update_fret_setting_in_metadata('donor_lifetime', donor_lifetime)
            except ValueError:
                pass
            
            try:
                if self.frequency_input.text():
                    frequency = float(self.frequency_input.text().strip())
                    self._update_fret_setting_in_metadata('frequency', frequency)
            except ValueError:
                pass
            
            self._update_fret_setting_in_metadata('donor_background', self.donor_background)
            self._update_fret_setting_in_metadata('background_real', self.background_real)
            self._update_fret_setting_in_metadata('background_imag', self.background_imag)
            self._update_fret_setting_in_metadata('donor_fretting_proportion', self.donor_fretting_proportion)
            self._update_fret_setting_in_metadata('use_colormap', self.use_colormap)
            self._update_fret_setting_in_metadata('background_positions_by_harmonic', self.background_positions_by_harmonic)
            self._update_fret_setting_in_metadata('analysis_performed', True)

        sample_layer = self.viewer.layers[labels_layer_name]
        phasor_data = (
            self.parent_widget._labels_layer_with_phasor_features.features
        )
        harmonic_mask = phasor_data['harmonic'] == self.parent_widget.harmonic
        real = phasor_data.loc[harmonic_mask, 'G']
        imag = phasor_data.loc[harmonic_mask, 'S']

        if not hasattr(self, 'donor_lifetime') or self.donor_lifetime is None:
            try:
                self.donor_lifetime = float(self.donor_line_edit.text())
            except ValueError:
                show_error("Please enter a valid donor lifetime")
                return

        if hasattr(self.current_donor_line, 'get_xydata'):
            neighbor_real, neighbor_imag = (
                self.current_donor_line.get_xydata().T
            )
        else:
            base_frequency = float(self.frequency_input.text().strip())
            effective_frequency = base_frequency * self.parent_widget.harmonic

            donor_trajectory_real, donor_trajectory_imag = (
                phasor_from_fret_donor(
                    effective_frequency,
                    self.donor_lifetime,
                    fret_efficiency=self._fret_efficiencies,
                    donor_background=self.donor_background,
                    background_imag=self.background_imag,
                    background_real=self.background_real,
                    donor_fretting=self.donor_fretting_proportion,
                )
            )
            neighbor_real, neighbor_imag = (
                donor_trajectory_real,
                donor_trajectory_imag,
            )

        fret_efficiency = phasor_nearest_neighbor(
            np.array(real),
            np.array(imag),
            neighbor_real,
            neighbor_imag,
            values=self._fret_efficiencies,
        )
        fret_efficiency = fret_efficiency.reshape(
            self.parent_widget._labels_layer_with_phasor_features.data.shape
        )

        fret_layer_name = f"FRET efficiency: {labels_layer_name}"
        
        default_colormap = 'viridis'
        default_contrast_limits = (0, 1)
        
        if hasattr(self, '_saved_colormap_name') and not self._updating_settings:
            if self._saved_colormap_colors is not None:
                from napari.utils.colormaps import Colormap
                if isinstance(self._saved_colormap_colors, list):
                    saved_colors = np.array(self._saved_colormap_colors)
                else:
                    saved_colors = self._saved_colormap_colors
                default_colormap = Colormap(colors=saved_colors, name="saved_custom")
            else:
                default_colormap = self._saved_colormap_name
            
            if isinstance(self._saved_contrast_limits, list):
                default_contrast_limits = tuple(self._saved_contrast_limits)
            else:
                default_contrast_limits = self._saved_contrast_limits
        
        selected_fret_layer = Image(
            fret_efficiency,
            name=fret_layer_name,
            scale=self.parent_widget._labels_layer_with_phasor_features.scale,
            colormap=default_colormap,
            contrast_limits=default_contrast_limits,
        )

        if fret_layer_name in self.viewer.layers:
            self.viewer.layers.remove(self.viewer.layers[fret_layer_name])

        self.fret_layer = self.viewer.add_layer(selected_fret_layer)
        self.fret_colormap = self.fret_layer.colormap.colors
        self.fret_layer.events.colormap.connect(self._on_colormap_changed)
        self.colormap_contrast_limits = self.fret_layer.contrast_limits
        self.fret_layer.events.contrast_limits.connect(
            self._on_contrast_limits_changed
        )

        if not hasattr(self, '_saved_colormap_name') or self._updating_settings:
            self._update_fret_setting_in_metadata('colormap_settings.colormap_name', 'viridis')
            self._update_fret_setting_in_metadata('colormap_settings.colormap_colors', None)
            self._update_fret_setting_in_metadata('colormap_settings.contrast_limits', [0, 1])
            self._update_fret_setting_in_metadata('colormap_settings.colormap_changed', False)

        self.plot_donor_trajectory()

        update_frequency_in_metadata(
            sample_layer,
            self.frequency_input.text().strip(),
        )
