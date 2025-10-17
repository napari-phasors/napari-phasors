from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from napari.layers import Image
from napari.utils.notifications import show_error, show_info
from phasorpy.lifetime import phasor_from_lifetime, polar_from_reference_phasor
from phasorpy.phasor import phasor_center, phasor_transform
from qtpy import uic
from qtpy.QtWidgets import QScrollArea, QVBoxLayout, QWidget

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

    def _populate_comboboxes(self):
        """Populate calibration layer combobox with image layers."""
        self.calibration_widget.calibration_layer_combobox.clear()
        image_layers = [
            layer for layer in self.viewer.layers if isinstance(layer, Image)
        ]
        for layer in image_layers:
            self.calibration_widget.calibration_layer_combobox.addItem(
                layer.name
            )

    def _update_button_state(self):
        """Update button text and state based on current layer's calibration status."""
        sample_name = (
            self.parent_widget.image_layer_with_phasor_features_combobox.currentText()
        )
        if sample_name == "" or sample_name not in self.viewer.layers:
            self.calibration_widget.calibrate_push_button.setText("Calibrate")
            return

        sample_metadata = self.viewer.layers[sample_name].metadata
        if (
            "settings" in sample_metadata.keys()
            and "calibrated" in sample_metadata["settings"].keys()
            and sample_metadata["settings"]["calibrated"] is True
        ):
            self.calibration_widget.calibrate_push_button.setText(
                "Uncalibrate"
            )
        else:
            self.calibration_widget.calibrate_push_button.setText("Calibrate")

    def _on_click(self):
        """Handle calibration button click."""
        sample_name = (
            self.parent_widget.image_layer_with_phasor_features_combobox.currentText()
        )
        calibration_name = (
            self.calibration_widget.calibration_layer_combobox.currentText()
        )

        if sample_name == "" or calibration_name == "":
            show_error("Select sample and calibration layers")
            return

        sample_layer = self.viewer.layers[sample_name]

        if self._is_layer_calibrated(sample_layer):
            self._uncalibrate_layer(sample_name)
        else:
            self._calibrate_layer(sample_name, calibration_name)

    def _is_layer_calibrated(self, sample_layer):
        """Check if a layer is already calibrated."""
        settings = sample_layer.metadata.get("settings", {})
        return settings.get("calibrated", False)

    def _calibrate_layer(self, sample_name, calibration_name):
        """Calibrate a layer using the specified calibration layer."""
        sample_layer = self.viewer.layers[sample_name]
        calibration_layer = self.viewer.layers[calibration_name]

        # Validate inputs
        frequency, lifetime = self._get_and_validate_inputs()
        if frequency is None or lifetime is None:
            return

        # Get phasor data and validate harmonics
        sample_phasor_data, harmonics = self._get_phasor_data(sample_layer)
        calibration_phasor_data, calibration_harmonics = self._get_phasor_data(
            calibration_layer
        )

        if not np.array_equal(harmonics, calibration_harmonics):
            show_error(
                "Harmonics in sample and calibration layers do not match"
            )
            return

        # Calculate calibration parameters
        phi_zero, mod_zero = self._calculate_calibration_parameters(
            calibration_layer,
            calibration_phasor_data,
            calibration_harmonics,
            frequency,
            harmonics,
            lifetime,
        )

        # Store calibration parameters
        settings = sample_layer.metadata.setdefault("settings", {})
        settings["calibration_phase"] = phi_zero.tolist()
        settings["calibration_modulation"] = mod_zero.tolist()
        settings["calibrated"] = True

        # Apply calibration transformation
        self._apply_phasor_transformation(sample_name, phi_zero, mod_zero)

        # Apply existing filters and thresholds
        self._apply_existing_filters_and_thresholds(sample_layer)

        show_info(f"Calibrated {sample_name}")
        self._update_button_state()
        self.parent_widget.plot()

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

        show_info(f"Uncalibrated {sample_name}")
        self._update_button_state()
        self.parent_widget.plot()

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
        phasor_data = layer.metadata["phasor_features_labels_layer"].features
        harmonics = np.unique(phasor_data["harmonic"])
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

        calibration_g = np.reshape(
            calibration_phasor_data["G_original"],
            (len(calibration_harmonics),) + calibration_mean.shape,
        )
        calibration_s = np.reshape(
            calibration_phasor_data["S_original"],
            (len(calibration_harmonics),) + calibration_mean.shape,
        )

        _, measured_re, measured_im = phasor_center(
            calibration_mean, calibration_g, calibration_s
        )

        known_re, known_im = phasor_from_lifetime(
            frequency * harmonics, lifetime
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
        sample_metadata = self.viewer.layers[sample_name].metadata
        sample_phasor_data = sample_metadata[
            "phasor_features_labels_layer"
        ].features
        harmonics = np.unique(sample_phasor_data["harmonic"])
        original_mean_shape = sample_metadata["original_mean"].shape

        # Handle multi-harmonic case
        axis = None
        if len(harmonics) > 1:
            axis = tuple(range(1, len(original_mean_shape) + 1))

        # Expand dimensions if needed
        if np.ndim(phi_zero) > 0 and axis is not None:
            phi_zero = np.expand_dims(phi_zero, axis=axis)
            mod_zero = np.expand_dims(mod_zero, axis=axis)

        # Transform original phasor data
        real_original, imag_original = phasor_transform(
            np.reshape(
                sample_phasor_data["G_original"],
                (len(harmonics),) + original_mean_shape,
            ),
            np.reshape(
                sample_phasor_data["S_original"],
                (len(harmonics),) + original_mean_shape,
            ),
            phi_zero,
            mod_zero,
        )

        # Transform current phasor data
        real, imag = phasor_transform(
            np.reshape(
                sample_phasor_data["G"],
                (len(harmonics),) + original_mean_shape,
            ),
            np.reshape(
                sample_phasor_data["S"],
                (len(harmonics),) + original_mean_shape,
            ),
            phi_zero,
            mod_zero,
        )

        # Update the phasor data
        sample_phasor_data["G_original"] = real_original.flatten()
        sample_phasor_data["S_original"] = imag_original.flatten()
        sample_phasor_data["G"] = real.flatten()
        sample_phasor_data["S"] = imag.flatten()
