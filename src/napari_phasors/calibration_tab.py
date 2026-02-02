from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from napari.layers import Image
from napari.utils.notifications import show_error, show_info
from phasorpy.lifetime import phasor_from_lifetime, polar_from_reference_phasor
from phasorpy.phasor import phasor_center, phasor_transform
from qtpy import uic
from qtpy.QtWidgets import QMessageBox, QScrollArea, QVBoxLayout, QWidget

from ._utils import apply_filter_and_threshold

if TYPE_CHECKING:
    import napari


class CalibrationWidget(QWidget):
    """Widget to calibrate a FLIM image layer using a calibration image."""

    def __init__(self, viewer: "napari.viewer.Viewer", parent=None):
        """Initialize the calibration widget."""
        super().__init__()
        self.viewer = viewer
        self.parent_widget = parent

        # Creates and empty widget
        self.calibration_widget = QWidget()
        uic.loadUi(
            Path(__file__).parent / "ui/calibration_widget.ui",
            self.calibration_widget,
        )

        # Connect callbacks
        self.calibration_widget.calibrate_push_button.clicked.connect(
            self._on_click
        )

        # Connect layer events to populate combobox and update button state
        self.viewer.layers.events.inserted.connect(self._populate_comboboxes)
        self.viewer.layers.events.removed.connect(self._populate_comboboxes)

        # Connect to update button state when layer selection changes
        if hasattr(
            self.parent_widget, 'image_layer_with_phasor_features_combobox'
        ):
            self.parent_widget.image_layer_with_phasor_features_combobox.currentTextChanged.connect(
                self._update_button_state
            )

        # Populate combobox
        self._populate_comboboxes()

        # Create scroll area and add calibration widget to it
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.calibration_widget)

        mainLayout = QVBoxLayout()
        mainLayout.addWidget(scroll_area)
        self.setLayout(mainLayout)

    def _populate_comboboxes(self, event=None):
        """Populate calibration layer combobox with image layers."""
        if getattr(self, '_populating_comboboxes', False):
            return

        self._populating_comboboxes = True

        try:
            current_text = (
                self.calibration_widget.calibration_layer_combobox.currentText()
            )

            self.calibration_widget.calibration_layer_combobox.blockSignals(
                True
            )

            self.calibration_widget.calibration_layer_combobox.clear()
            image_layers = [
                layer
                for layer in self.viewer.layers
                if isinstance(layer, Image)
                and 'G' in layer.metadata
                and 'S' in layer.metadata
            ]

            layer_names = [layer.name for layer in image_layers]
            self.calibration_widget.calibration_layer_combobox.addItems(
                layer_names
            )

            if current_text in layer_names:
                index = self.calibration_widget.calibration_layer_combobox.findText(
                    current_text
                )
                if index >= 0:
                    self.calibration_widget.calibration_layer_combobox.setCurrentIndex(
                        index
                    )

            self.calibration_widget.calibration_layer_combobox.blockSignals(
                False
            )

            for layer in image_layers:
                try:
                    layer.events.name.disconnect(self._populate_comboboxes)
                except (TypeError, ValueError):
                    pass
                layer.events.name.connect(self._populate_comboboxes)

        finally:
            self._populating_comboboxes = False

    def _on_image_layer_changed(self):
        """Update button state when the selected image layer changes."""
        self._update_button_state()

    def _update_button_state(self):
        """Update button text and state based on current layer's calibration status."""
        selected_layers = self.parent_widget.get_selected_layers()
        if not selected_layers:
            self.calibration_widget.calibrate_push_button.setText("Calibrate")
            return

        # Check if any selected layer is calibrated
        any_calibrated = any(self._is_layer_calibrated(layer) for layer in selected_layers)
        
        if any_calibrated:
            self.calibration_widget.calibrate_push_button.setText(
                "Uncalibrate"
            )
        else:
            self.calibration_widget.calibrate_push_button.setText("Calibrate")

    def _on_click(self):
        """Handle calibration button click for all selected layers."""
        selected_layers = self.parent_widget.get_selected_layers()
        if not selected_layers:
            show_error("Select sample and calibration layers")
            return
            
        calibration_name = (
            self.calibration_widget.calibration_layer_combobox.currentText()
        )

        if calibration_name == "":
            show_error("Select sample and calibration layers")
            return

        any_calibrated = any(self._is_layer_calibrated(layer) for layer in selected_layers)

        if any_calibrated:
            calibrated_layers = [layer for layer in selected_layers if self._is_layer_calibrated(layer)]
            for layer in calibrated_layers:
                self._uncalibrate_layer(layer.name)
            
            if calibrated_layers:
                layer_names = ", ".join([layer.name for layer in calibrated_layers])
                show_info(f"Uncalibrated {layer_names}")
        else:
            calibrated_layers = []
            for layer in selected_layers:
                result = self._calibrate_layer(layer.name, calibration_name)
                if result is not False:
                    calibrated_layers.append(layer)
            
            if calibrated_layers:
                layer_names = ", ".join([layer.name for layer in calibrated_layers])
                show_info(f"Calibrated {layer_names}")
        
        self._update_button_state()
        self.parent_widget.plot()

    def _is_layer_calibrated(self, sample_layer):
        """Check if a layer is already calibrated."""
        settings = sample_layer.metadata.get("settings", {})
        return settings.get("calibrated", False)

    def _calibrate_layer(self, sample_name, calibration_name):
        """Calibrate a layer using the specified calibration layer."""
        sample_layer = self.viewer.layers[sample_name]
        calibration_layer = self.viewer.layers[calibration_name]

        calibration_was_calibrated = False
        if self._is_layer_calibrated(calibration_layer):
            reply = QMessageBox.question(
                self,
                'Calibration Layer Already Calibrated',
                f'The calibration layer "{calibration_name}" is already calibrated.\n\n'
                'Would you like to use the uncalibrated data as reference?\n\n'
                'Yes: Use original uncalibrated data\n'
                'No: Use current calibrated data',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            
            if reply == QMessageBox.Yes:
                calibration_was_calibrated = True
                self._uncalibrate_layer(calibration_name)
                calibration_layer = self.viewer.layers[calibration_name]

        frequency, lifetime = self._get_and_validate_inputs()
        if frequency is None or lifetime is None:
            if calibration_was_calibrated:
                self._restore_calibration(calibration_name)
            return False

        sample_phasor_data, harmonics = self._get_phasor_data(sample_layer)
        calibration_phasor_data, calibration_harmonics = self._get_phasor_data(
            calibration_layer
        )

        if not np.array_equal(harmonics, calibration_harmonics):
            show_error(
                "Harmonics in sample and calibration layers do not match"
            )
            if calibration_was_calibrated:
                self._restore_calibration(calibration_name)
            return False

        phi_zero, mod_zero = self._calculate_calibration_parameters(
            calibration_layer,
            calibration_phasor_data,
            calibration_harmonics,
            frequency,
            harmonics,
            lifetime,
        )

        settings = sample_layer.metadata.setdefault("settings", {})
        settings["calibration_phase"] = phi_zero.tolist()
        settings["calibration_modulation"] = mod_zero.tolist()
        settings["calibrated"] = True

        self._apply_phasor_transformation(sample_name, phi_zero, mod_zero)

        self._apply_existing_filters_and_thresholds(sample_layer)

        if calibration_was_calibrated:
            self._restore_calibration(calibration_name)

    def _restore_calibration(self, layer_name):
        """Restore calibration to a layer using stored parameters."""
        layer = self.viewer.layers[layer_name]
        settings = layer.metadata.get("settings", {})
        
        phi_zero = settings.get("calibration_phase")
        mod_zero = settings.get("calibration_modulation")
        
        if phi_zero is not None and mod_zero is not None:
            if isinstance(phi_zero, list):
                phi_zero = np.array(phi_zero)
            if isinstance(mod_zero, list):
                mod_zero = np.array(mod_zero)
            
            self._apply_phasor_transformation(layer_name, phi_zero, mod_zero)
            settings["calibrated"] = True
            self._apply_existing_filters_and_thresholds(layer)

    def _uncalibrate_layer(self, sample_name):
        """Uncalibrate a layer."""
        if sample_name == "":
            return

        sample_layer = self.viewer.layers[sample_name]
        sample_metadata = sample_layer.metadata
        settings = sample_metadata.get("settings", {})

        phi_zero = settings.get("calibration_phase")
        mod_zero = settings.get("calibration_modulation")

        if phi_zero is None or mod_zero is None:
            show_error("Layer is not calibrated")
            return

        phi_zero_inv, mod_zero_inv = self._invert_calibration_parameters(
            phi_zero, mod_zero
        )

        self._apply_phasor_transformation(
            sample_name, phi_zero_inv, mod_zero_inv
        )

        settings["calibrated"] = False
        settings.pop("calibration_phase", None)
        settings.pop("calibration_modulation", None)

        self._apply_existing_filters_and_thresholds(sample_layer)

    def _get_and_validate_inputs(self):
        """Get and validate frequency and lifetime inputs."""
        frequency = self.calibration_widget.frequency_input.text().strip()
        lifetime = (
            self.calibration_widget.lifetime_line_edit_widget.text().strip()
        )

        if frequency == "":
            show_error("Enter frequency")
            return None, None
        if lifetime == "":
            show_error("Enter reference lifetime")
            return None, None

        return float(frequency), float(lifetime)

    def _get_phasor_data(self, layer):
        """Get phasor data and harmonics from a layer."""
        phasor_data = {
            "G_original": layer.metadata.get("G_original"),
            "S_original": layer.metadata.get("S_original"),
            "G": layer.metadata.get("G"),
            "S": layer.metadata.get("S"),
        }
        harmonics = layer.metadata.get("harmonics")
        return phasor_data, harmonics

    def _calculate_calibration_parameters(
        self,
        calibration_layer,
        calibration_phasor_data,
        calibration_harmonics,
        frequency,
        harmonics,
        lifetime,
    ):
        """Calculate calibration phase and modulation parameters."""
        calibration_mean = calibration_layer.metadata["original_mean"]

        calibration_g = calibration_phasor_data["G_original"]
        calibration_s = calibration_phasor_data["S_original"]

        _, measured_re, measured_im = phasor_center(
            calibration_mean, calibration_g, calibration_s
        )

        harmonics_array = np.atleast_1d(harmonics)

        known_re, known_im = phasor_from_lifetime(
            frequency * harmonics_array, lifetime
        )
        phi_zero, mod_zero = polar_from_reference_phasor(
            measured_re, measured_im, known_re, known_im
        )

        return phi_zero, mod_zero

    def _invert_calibration_parameters(self, phi_zero, mod_zero):
        """Invert calibration parameters for uncalibration."""
        if isinstance(phi_zero, list):
            phi_zero = np.array(phi_zero)
        if isinstance(mod_zero, list):
            mod_zero = np.array(mod_zero)

        if np.ndim(phi_zero) > 0:
            phi_zero_inv = -phi_zero.copy()
            mod_zero_inv = 1.0 / mod_zero.copy()
        else:
            phi_zero_inv = -phi_zero
            mod_zero_inv = 1.0 / mod_zero

        return phi_zero_inv, mod_zero_inv

    def _apply_existing_filters_and_thresholds(self, sample_layer):
        """Apply existing filter and threshold settings if they exist."""
        settings = sample_layer.metadata.get("settings", {})

        filter_settings = settings.get("filter", {})
        filter_size = filter_settings.get("size")
        filter_repeat = filter_settings.get("repeat")

        threshold = settings.get("threshold")

        if (
            filter_size is not None
            or filter_repeat is not None
            or threshold is not None
        ):
            apply_filter_and_threshold(
                sample_layer,
                threshold=threshold,
                size=filter_size,
                repeat=filter_repeat,
            )

    def _apply_phasor_transformation(self, sample_name, phi_zero, mod_zero):
        """Apply phasor transformation with given correction parameters."""
        sample_layer = self.viewer.layers[sample_name]
        sample_metadata = sample_layer.metadata

        harmonics = sample_metadata.get("harmonics")
        g_original = sample_metadata["G_original"]
        s_original = sample_metadata["S_original"]
        g_current = sample_metadata["G"]
        s_current = sample_metadata["S"]

        if isinstance(phi_zero, list):
            phi_zero = np.array(phi_zero)
        if isinstance(mod_zero, list):
            mod_zero = np.array(mod_zero)

        harmonics = np.atleast_1d(harmonics)

        if g_original.ndim > 1 and len(harmonics) > 1:
            spatial_dims = g_original.ndim - 1
            expand_shape = (slice(None),) + (None,) * spatial_dims

            if np.ndim(phi_zero) > 0:
                phi_zero_expanded = phi_zero[expand_shape]
                mod_zero_expanded = mod_zero[expand_shape]
            else:
                phi_zero_expanded = phi_zero
                mod_zero_expanded = mod_zero
        else:
            phi_zero_expanded = phi_zero
            mod_zero_expanded = mod_zero

        real_original, imag_original = phasor_transform(
            g_original,
            s_original,
            phi_zero_expanded,
            mod_zero_expanded,
        )

        real, imag = phasor_transform(
            g_current,
            s_current,
            phi_zero_expanded,
            mod_zero_expanded,
        )

        sample_metadata["G_original"] = real_original
        sample_metadata["S_original"] = imag_original
        sample_metadata["G"] = real
        sample_metadata["S"] = imag

        if self.parent_widget:
            self.parent_widget.refresh_phasor_data()
