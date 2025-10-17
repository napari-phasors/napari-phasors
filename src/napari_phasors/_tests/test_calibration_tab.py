from unittest.mock import Mock, patch

import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal
from phasorpy.lifetime import (
    phasor_calibrate,
    phasor_from_lifetime,
    polar_from_reference_phasor,
)
from phasorpy.phasor import phasor_center

from napari_phasors._tests.test_plotter import create_image_layer_with_phasors
from napari_phasors.plotter import PlotterWidget


def test_calibration_widget_initialization(make_napari_viewer):
    """Test the initialization of the CalibrationWidget."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    widget = parent.calibration_tab

    # Basic widget structure tests
    assert widget.viewer == viewer
    assert widget.parent_widget == parent
    assert widget.layout().count() > 0

    # Test initial UI state
    assert widget.calibration_widget.frequency_input.text() == ""
    assert widget.calibration_widget.lifetime_line_edit_widget.text() == ""
    assert (
        widget.calibration_widget.calibrate_push_button.text() == "Calibrate"
    )

    # Test combobox initialization (should be empty initially)
    assert widget.calibration_widget.calibration_layer_combobox.count() == 0


def test_calibration_widget_populate_comboboxes(make_napari_viewer):
    """Test that comboboxes are populated with image layers."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    widget = parent.calibration_tab

    # Initially empty
    assert widget.calibration_widget.calibration_layer_combobox.count() == 0

    # Add image layers
    sample_layer = create_image_layer_with_phasors()
    sample_layer.name = "sample_layer"
    calibration_layer = create_image_layer_with_phasors()
    calibration_layer.name = "calibration_layer"

    viewer.add_layer(sample_layer)
    viewer.add_layer(calibration_layer)

    # Check combobox is populated
    combobox = widget.calibration_widget.calibration_layer_combobox
    assert combobox.count() == 2
    layer_names = [combobox.itemText(i) for i in range(combobox.count())]
    assert "sample_layer" in layer_names
    assert "calibration_layer" in layer_names


def test_calibration_widget_layer_events(make_napari_viewer):
    """Test that widget responds to layer addition/removal events."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    widget = parent.calibration_tab

    # Add a layer
    test_layer = create_image_layer_with_phasors()
    test_layer.name = "test_layer"
    viewer.add_layer(test_layer)

    # Check combobox updated
    combobox = widget.calibration_widget.calibration_layer_combobox
    assert combobox.count() == 1
    assert combobox.itemText(0) == "test_layer"

    # Remove the layer
    viewer.layers.remove("test_layer")

    # Check combobox updated
    assert combobox.count() == 0


def test_calibration_click_no_layers_selected(make_napari_viewer):
    """Test calibration click with no layers selected."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    widget = parent.calibration_tab

    with patch("napari_phasors.calibration_tab.show_error") as mock_show_error:
        widget._on_click()
        mock_show_error.assert_called_once_with(
            "Select sample and calibration layers"
        )


def test_calibration_click_missing_frequency(make_napari_viewer):
    """Test calibration click with missing frequency."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)

    widget = parent.calibration_tab

    # Add layers
    sample_layer = create_image_layer_with_phasors()
    sample_layer.name = "sample_layer"
    calibration_layer = create_image_layer_with_phasors()
    calibration_layer.name = "calibration_layer"
    viewer.add_layer(sample_layer)
    viewer.add_layer(calibration_layer)

    # Set calibration layer but leave frequency empty
    widget.calibration_widget.calibration_layer_combobox.setCurrentText(
        "calibration_layer"
    )

    with patch("napari_phasors.calibration_tab.show_error") as mock_show_error:
        widget._on_click()
        mock_show_error.assert_called_once_with("Enter frequency")


def test_calibration_click_missing_lifetime(make_napari_viewer):
    """Test calibration click with missing reference lifetime."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)

    widget = parent.calibration_tab

    # Add layers
    sample_layer = create_image_layer_with_phasors()
    sample_layer.name = "sample_layer"
    calibration_layer = create_image_layer_with_phasors()
    calibration_layer.name = "calibration_layer"
    viewer.add_layer(sample_layer)
    viewer.add_layer(calibration_layer)

    # Set parameters but leave lifetime empty
    widget.calibration_widget.calibration_layer_combobox.setCurrentText(
        "calibration_layer"
    )
    widget.calibration_widget.frequency_input.setText("80")

    with patch("napari_phasors.calibration_tab.show_error") as mock_show_error:
        widget._on_click()
        mock_show_error.assert_called_once_with("Enter reference lifetime")


def test_calibration_button_state_updates(make_napari_viewer):
    """Test that calibration button text updates based on layer calibration status."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    parent.image_layer_with_phasor_features_combobox = Mock()

    widget = parent.calibration_tab

    # Test with no layer selected
    parent.image_layer_with_phasor_features_combobox.currentText.return_value = (
        ""
    )
    widget._update_button_state()
    assert (
        widget.calibration_widget.calibrate_push_button.text() == "Calibrate"
    )

    # Test with uncalibrated layer
    sample_layer = create_image_layer_with_phasors()
    sample_layer.name = "sample_layer"
    viewer.add_layer(sample_layer)
    parent.image_layer_with_phasor_features_combobox.currentText.return_value = (
        "sample_layer"
    )

    widget._update_button_state()
    assert (
        widget.calibration_widget.calibrate_push_button.text() == "Calibrate"
    )

    # Test with calibrated layer
    sample_layer.metadata["settings"] = {"calibrated": True}
    widget._update_button_state()
    assert (
        widget.calibration_widget.calibrate_push_button.text() == "Uncalibrate"
    )


def test_calibrate_layer_success(make_napari_viewer):
    """Test calibrating a layer."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    widget = parent.calibration_tab

    # Add uncalibrated layer
    sample_layer = create_image_layer_with_phasors()
    sample_layer.name = "sample_layer"
    viewer.add_layer(sample_layer)

    widget.calibration_widget.calibration_layer_combobox.setCurrentText(
        "sample_layer"
    )
    widget.calibration_widget.frequency_input.setText("80")
    widget.calibration_widget.lifetime_line_edit_widget.setText("2.0")

    # Make copies of the original data before calibration modifies it
    original_image = sample_layer.data.copy()
    g_original = (
        sample_layer.metadata['phasor_features_labels_layer']
        .features['G']
        .values.copy()
    )
    s_original = (
        sample_layer.metadata['phasor_features_labels_layer']
        .features['S']
        .values.copy()
    )
    mean_original = sample_layer.metadata['original_mean'].copy()
    mean_shape = mean_original.shape
    harmonic = [1, 2, 3]
    frequency = [80 * h for h in harmonic]
    lifetime = 2.0
    g_reshaped = g_original.reshape((len(harmonic),) + mean_shape)
    s_reshaped = s_original.reshape((len(harmonic),) + mean_shape)

    # Calculate expected values using copies
    real, imag = phasor_calibrate(
        g_reshaped,
        s_reshaped,
        mean_original,
        g_reshaped,
        s_reshaped,
        frequency,
        lifetime,
    )
    _, real_center, imag_center = phasor_center(
        mean_original, g_reshaped, s_reshaped
    )
    known_re, known_im = phasor_from_lifetime(frequency, lifetime)
    phi, mod = polar_from_reference_phasor(
        real_center, imag_center, known_re, known_im
    )

    # Click Calibrate button (this will modify the layer's data)
    widget.calibration_widget.calibrate_push_button.click()

    assert sample_layer.metadata["settings"]["calibrated"] is True
    assert_array_equal(
        sample_layer.metadata["settings"]["calibration_phase"], phi
    )
    assert_array_equal(
        sample_layer.metadata["settings"]["calibration_modulation"], mod
    )
    assert_array_equal(
        sample_layer.metadata['phasor_features_labels_layer'].features['G'],
        real.flatten(),
    )
    assert_array_equal(
        sample_layer.metadata['phasor_features_labels_layer'].features['S'],
        imag.flatten(),
    )
    assert_array_equal(sample_layer.metadata['original_mean'], mean_original)
    assert_array_equal(sample_layer.data, original_image)


def test_uncalibrate_layer_not_calibrated(make_napari_viewer):
    """Test uncalibrating a layer that is not calibrated."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    widget = parent.calibration_tab

    # Add uncalibrated layer
    sample_layer = create_image_layer_with_phasors()
    sample_layer.name = "sample_layer"
    viewer.add_layer(sample_layer)

    with patch("napari_phasors.calibration_tab.show_error") as mock_show_error:
        widget._uncalibrate_layer("sample_layer")
        mock_show_error.assert_called_once_with("Layer is not calibrated")


def test_uncalibrate_layer_missing_modulation(make_napari_viewer):
    """Test uncalibrating a layer with missing calibration modulation."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    widget = parent.calibration_tab

    # Add layer with phase but no modulation
    sample_layer = create_image_layer_with_phasors()
    sample_layer.name = "sample_layer"
    sample_layer.metadata["settings"] = {
        "calibrated": True,
        "calibration_phase": 0.5,
    }
    viewer.add_layer(sample_layer)

    with patch("napari_phasors.calibration_tab.show_error") as mock_show_error:
        widget._uncalibrate_layer("sample_layer")
        mock_show_error.assert_called_once_with("Layer is not calibrated")


def test_uncalibrate_layer_success(make_napari_viewer):
    """Test successful uncalibration of a layer."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    widget = parent.calibration_tab

    # Add uncalibrated layer
    sample_layer = create_image_layer_with_phasors()
    sample_layer.name = "sample_layer"
    viewer.add_layer(sample_layer)

    widget.calibration_widget.calibration_layer_combobox.setCurrentText(
        "sample_layer"
    )
    widget.calibration_widget.frequency_input.setText("80")
    widget.calibration_widget.lifetime_line_edit_widget.setText("2.0")

    # Make copies of the original data before calibration modifies it
    original_image = sample_layer.data.copy()
    g_original = (
        sample_layer.metadata['phasor_features_labels_layer']
        .features['G']
        .values.copy()
    )
    s_original = (
        sample_layer.metadata['phasor_features_labels_layer']
        .features['S']
        .values.copy()
    )
    mean_original = sample_layer.metadata['original_mean'].copy()

    # Click Calibrate button
    widget.calibration_widget.calibrate_push_button.click()

    assert sample_layer.metadata["settings"]["calibrated"] is True

    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(
            g_original,
            sample_layer.metadata['phasor_features_labels_layer'].features[
                'G'
            ],
        )
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(
            s_original,
            sample_layer.metadata['phasor_features_labels_layer'].features[
                'S'
            ],
        )
    assert_array_equal(mean_original, sample_layer.metadata['original_mean'])
    assert_array_equal(sample_layer.data, original_image)

    # Click Uncalibrate button
    widget.calibration_widget.calibrate_push_button.click()

    assert sample_layer.metadata["settings"]["calibrated"] is False
    assert "calibration_phase" not in sample_layer.metadata["settings"]
    assert "calibration_modulation" not in sample_layer.metadata["settings"]
    assert_almost_equal(
        sample_layer.metadata['phasor_features_labels_layer'].features['G'],
        g_original,
    )
    assert_almost_equal(
        sample_layer.metadata['phasor_features_labels_layer'].features['S'],
        s_original,
    )
    assert_almost_equal(sample_layer.metadata['original_mean'], mean_original)
    assert_almost_equal(sample_layer.data, original_image)


def test_uncalibrate_layer_empty_name(make_napari_viewer):
    """Test uncalibrating with empty layer name."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    widget = parent.calibration_tab

    result = widget._uncalibrate_layer("")
    assert result is None


def test_harmonic_mismatch_error(make_napari_viewer):
    """Test error when sample and calibration harmonics don't match."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)

    widget = parent.calibration_tab

    # Add layers with different harmonics
    sample_layer = create_image_layer_with_phasors()
    sample_layer.name = "sample_layer"
    calibration_layer = create_image_layer_with_phasors()
    calibration_layer.name = "calibration_layer"

    # Modify harmonics to be different
    sample_features = sample_layer.metadata[
        "phasor_features_labels_layer"
    ].features
    calibration_features = calibration_layer.metadata[
        "phasor_features_labels_layer"
    ].features
    calibration_features["harmonic"] = (
        calibration_features["harmonic"] + 1
    )  # Make them different

    viewer.add_layer(sample_layer)
    viewer.add_layer(calibration_layer)

    # Set up UI inputs
    widget.calibration_widget.calibration_layer_combobox.setCurrentText(
        "calibration_layer"
    )
    widget.calibration_widget.frequency_input.setText("80")
    widget.calibration_widget.lifetime_line_edit_widget.setText("2")

    with patch("napari_phasors.calibration_tab.show_error") as mock_show_error:
        widget._on_click()
        mock_show_error.assert_called_once_with(
            "Harmonics in sample and calibration layers do not match"
        )


def test_on_image_layer_changed_with_frequency(make_napari_viewer):
    """Test that frequency is populated when image layer changes."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)

    widget = parent.calibration_tab

    # Add layer with frequency in metadata
    test_layer = create_image_layer_with_phasors()
    test_layer.name = "test_layer"
    test_layer.metadata["settings"] = {"frequency": 80}
    viewer.add_layer(test_layer)

    parent._sync_frequency_inputs_from_metadata()
    assert widget.calibration_widget.frequency_input.text() == "80.0"
