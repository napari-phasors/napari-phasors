from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from napari.layers import Image
from napari.utils.notifications import show_error, show_info
from phasorpy.phasor import (
    phasor_calibrate,
    phasor_center,
    phasor_from_lifetime,
    phasor_transform,
    polar_from_reference_phasor,
)
from qtpy import uic
from qtpy.QtWidgets import QVBoxLayout, QWidget

from ._utils import update_frequency_in_metadata

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

        # Connect layer events to populate combobox
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

        mainLayout = QVBoxLayout()
        mainLayout.addWidget(self.calibration_widget)
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

    def _on_image_layer_changed(self):
        """Handle changes to the image layer selection."""
        layer_name = (
            self.parent_widget.image_layer_with_phasor_features_combobox.currentText()
        )
        if layer_name == "":
            return
        layer_metadata = self.viewer.layers[layer_name].metadata
        if (
            'settings' in layer_metadata.keys()
            and 'frequency' in layer_metadata['settings'].keys()
        ):
            self.calibration_widget.frequency_input.setText(
                str(layer_metadata['settings']['frequency'])
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

        sample_metadata = self.viewer.layers[sample_name].metadata

        # Check if layer is already calibrated
        if (
            "settings" in sample_metadata.keys()
            and "calibrated" in sample_metadata["settings"].keys()
            and sample_metadata["settings"]["calibrated"] is True
        ):
            # Uncalibrate functionality
            self._uncalibrate_layer(sample_name)
            return

        # Continue with calibration process
        frequency = self.calibration_widget.frequency_input.text().strip()
        lifetime = (
            self.calibration_widget.lifetime_line_edit_widget.text().strip()
        )
        if frequency == "":
            show_error("Enter frequency")
            return
        if lifetime == "":
            show_error("Enter reference lifetime")
            return
        frequency = float(frequency)
        update_frequency_in_metadata(self.parent_widget, frequency)
        lifetime = float(lifetime)
        sample_phasor_data = sample_metadata[
            "phasor_features_labels_layer"
        ].features
        harmonics = np.unique(sample_phasor_data["harmonic"])
        original_mean_shape = sample_metadata["original_mean"].shape
        calibration_phasor_data = (
            self.viewer.layers[calibration_name]
            .metadata["phasor_features_labels_layer"]
            .features
        )
        calibration_mean = self.viewer.layers[calibration_name].metadata[
            "original_mean"
        ]
        calibration_harmonics = np.unique(calibration_phasor_data["harmonic"])
        if not np.array_equal(harmonics, calibration_harmonics):
            show_error(
                "Harmonics in sample and calibration layers do not match"
            )
            return
        calibration_g = np.reshape(
            calibration_phasor_data["G_original"],
            (len(calibration_harmonics),) + calibration_mean.shape,
        )
        calibration_s = np.reshape(
            calibration_phasor_data["S_original"],
            (len(calibration_harmonics),) + calibration_mean.shape,
        )
        if "settings" not in sample_metadata.keys():
            sample_metadata["settings"] = {}

        # Perform calibration
        real_original, imag_original = phasor_calibrate(
            np.reshape(
                sample_phasor_data["G_original"],
                (len(harmonics),) + original_mean_shape,
            ),
            np.reshape(
                sample_phasor_data["S_original"],
                (len(harmonics),) + original_mean_shape,
            ),
            calibration_mean,
            calibration_g,
            calibration_s,
            frequency=frequency,
            lifetime=lifetime,
            harmonic=harmonics.tolist(),
        )
        real, imag = phasor_calibrate(
            np.reshape(
                sample_phasor_data["G"],
                (len(harmonics),) + original_mean_shape,
            ),
            np.reshape(
                sample_phasor_data["S"],
                (len(harmonics),) + original_mean_shape,
            ),
            calibration_mean,
            calibration_g,
            calibration_s,
            frequency=frequency,
            lifetime=lifetime,
            harmonic=harmonics.tolist(),
        )
        sample_phasor_data["G_original"] = real_original.flatten()
        sample_phasor_data["S_original"] = imag_original.flatten()
        sample_phasor_data["G"] = real.flatten()
        sample_phasor_data["S"] = imag.flatten()
        sample_metadata["settings"]["calibrated"] = True

        # First calculate the correction values manually to store them
        _, measured_re, measured_im = phasor_center(
            calibration_mean,
            calibration_g,
            calibration_s,
        )

        known_re, known_im = phasor_from_lifetime(
            frequency * harmonics, lifetime
        )

        # Get the phase and modulation correction values
        phi_zero, mod_zero = polar_from_reference_phasor(
            measured_re, measured_im, known_re, known_im
        )
        sample_metadata["settings"]["calibration_phase"] = phi_zero
        sample_metadata["settings"]["calibration_modulation"] = mod_zero

        show_info(f"Calibrated {sample_name}")
        self._update_button_state()  # Update button text after calibration
        self.parent_widget.plot()

    def _uncalibrate_layer(self, sample_name):
        """Uncalibrate a layer."""
        if sample_name == "":
            return
        sample_metadata = self.viewer.layers[sample_name].metadata
        sample_phasor_data = sample_metadata[
            "phasor_features_labels_layer"
        ].features
        phi_zero = sample_metadata.get("settings", {}).get(
            "calibration_phase", None
        )
        if phi_zero is None:
            show_error("Layer is not calibrated")
            return
        mod_zero = sample_metadata.get("settings", {}).get(
            "calibration_modulation", None
        )
        if mod_zero is None:
            show_error("Layer is not calibrated")
            return

        harmonics = np.unique(sample_phasor_data["harmonic"])
        original_mean_shape = sample_metadata["original_mean"].shape
        axis = None
        if len(harmonics) > 1:
            # axis is all dimensions except the first one
            axis = tuple(range(1, len(original_mean_shape) + 1))

        # Reset phasor features
        if np.ndim(phi_zero) > 0:
            np.negative(phi_zero, out=phi_zero)
            np.reciprocal(mod_zero, out=mod_zero)
            if axis is not None:
                phi_zero = np.expand_dims(phi_zero, axis=axis)
                mod_zero = np.expand_dims(mod_zero, axis=axis)
        else:
            phi_zero = -phi_zero
            mod_zero = 1.0 / mod_zero

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

        sample_phasor_data["G_original"] = real_original.flatten()
        sample_phasor_data["S_original"] = imag_original.flatten()
        sample_phasor_data["G"] = real.flatten()
        sample_phasor_data["S"] = imag.flatten()

        # Reset calibration status
        if "settings" in sample_metadata.keys():
            sample_metadata["settings"]["calibrated"] = False
            # Remove calibration parameters
            sample_metadata["settings"].pop("calibration_phase", None)
            sample_metadata["settings"].pop("calibration_modulation", None)

        show_info(f"Uncalibrated {sample_name}")
        self._update_button_state()  # Update button text after uncalibration
        self.parent_widget.plot()
