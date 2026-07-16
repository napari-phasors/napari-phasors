import contextlib
from unittest.mock import patch

import numpy as np
from matplotlib.figure import Figure
from napari.layers import Image
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
)
from superqt import QRangeSlider, QToggleSwitch

from napari_phasors._tests.test_plotter import create_image_layer_with_phasors
from napari_phasors.plotter import PlotterWidget


def test_filter_widget_initialization_values(make_viewer_model, qtbot):
    """Test the initialization of the Filter Widget."""
    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    # Basic widget structure tests
    assert filter_widget.viewer == viewer
    assert filter_widget.parent_widget == parent
    assert isinstance(filter_widget.layout(), QVBoxLayout)

    # Test initial attribute values
    assert filter_widget._phasors_selected_layer is None
    assert filter_widget.threshold_factor == 1
    assert filter_widget.threshold_line_lower is None
    assert filter_widget.threshold_line_upper is None
    assert filter_widget.threshold_area_lower is None
    assert filter_widget.threshold_area_upper is None

    # Test histogram figure initialization
    assert isinstance(filter_widget.hist_fig, Figure)
    assert filter_widget.hist_ax is not None

    # Test filter method combobox
    assert hasattr(filter_widget, 'filter_method_combobox')
    assert isinstance(filter_widget.filter_method_combobox, QComboBox)
    assert filter_widget.filter_method_combobox.count() == 3
    assert filter_widget.filter_method_combobox.itemText(0) == "None"
    assert filter_widget.filter_method_combobox.itemText(1) == "Median"
    assert (
        filter_widget.filter_method_combobox.itemText(2)
        == "Wavelet (binlet pawFLIM)"
    )
    assert filter_widget.filter_method_combobox.currentText() == "None"

    # Test threshold method combobox
    assert hasattr(filter_widget, 'threshold_method_combobox')
    assert isinstance(filter_widget.threshold_method_combobox, QComboBox)
    assert filter_widget.threshold_method_combobox.count() == 5
    assert filter_widget.threshold_method_combobox.itemText(0) == "None"
    assert filter_widget.threshold_method_combobox.itemText(1) == "Manual"
    assert filter_widget.threshold_method_combobox.itemText(2) == "Otsu"
    assert filter_widget.threshold_method_combobox.itemText(3) == "Li"
    assert filter_widget.threshold_method_combobox.itemText(4) == "Yen"
    assert filter_widget.threshold_method_combobox.currentText() == "None"

    # Test log scale checkbox
    assert hasattr(filter_widget, 'log_scale_checkbox')
    assert isinstance(filter_widget.log_scale_checkbox, QToggleSwitch)
    assert filter_widget.log_scale_checkbox.text() == "Log Scale Histogram"
    assert (
        not filter_widget.log_scale_checkbox.isChecked()
    )  # Should start unchecked

    # Test median filter UI components
    assert hasattr(filter_widget, 'median_filter_label')
    assert isinstance(filter_widget.median_filter_label, QLabel)
    assert filter_widget.median_filter_label.text() == "Kernel Size: 3 x 3"

    assert hasattr(filter_widget, 'median_filter_spinbox')
    assert isinstance(filter_widget.median_filter_spinbox, QSpinBox)
    assert filter_widget.median_filter_spinbox.minimum() == 2
    assert filter_widget.median_filter_spinbox.maximum() == 99
    assert filter_widget.median_filter_spinbox.value() == 3

    assert hasattr(filter_widget, 'median_filter_repetition_spinbox')
    assert isinstance(filter_widget.median_filter_repetition_spinbox, QSpinBox)
    assert filter_widget.median_filter_repetition_spinbox.minimum() == 1
    assert filter_widget.median_filter_repetition_spinbox.value() == 1

    # Test wavelet filter UI components
    assert hasattr(filter_widget, 'wavelet_sigma_spinbox')
    assert isinstance(filter_widget.wavelet_sigma_spinbox, QDoubleSpinBox)
    assert filter_widget.wavelet_sigma_spinbox.minimum() == 0.1
    assert filter_widget.wavelet_sigma_spinbox.maximum() == 10.0
    assert filter_widget.wavelet_sigma_spinbox.value() == 2.0

    assert hasattr(filter_widget, 'wavelet_levels_spinbox')
    assert isinstance(filter_widget.wavelet_levels_spinbox, QSpinBox)
    assert filter_widget.wavelet_levels_spinbox.minimum() == 1
    assert filter_widget.wavelet_levels_spinbox.maximum() == 10
    assert filter_widget.wavelet_levels_spinbox.value() == 1

    # Test warning label for harmonics
    assert hasattr(filter_widget, 'harmonic_warning_label')
    assert isinstance(filter_widget.harmonic_warning_label, QLabel)
    assert filter_widget.harmonic_warning_label.isHidden()

    # Test threshold editable text fields (min and max intensity)
    assert hasattr(filter_widget, 'min_threshold_edit')
    assert isinstance(filter_widget.min_threshold_edit, QLineEdit)
    assert filter_widget.min_threshold_edit.text() == "0.00"

    assert hasattr(filter_widget, 'max_threshold_edit')
    assert isinstance(filter_widget.max_threshold_edit, QLineEdit)
    assert filter_widget.max_threshold_edit.text() == "0.00"

    # Test threshold range slider
    assert hasattr(filter_widget, 'threshold_slider')
    assert isinstance(filter_widget.threshold_slider, QRangeSlider)
    assert filter_widget.threshold_slider.orientation() == Qt.Horizontal
    assert filter_widget.threshold_slider.minimum() == 0
    assert filter_widget.threshold_slider.maximum() == 100
    assert filter_widget.threshold_slider.value() == (0, 100)

    # Test apply button
    assert hasattr(filter_widget, 'apply_button')
    assert isinstance(filter_widget.apply_button, QPushButton)
    assert filter_widget.apply_button.text() == "Apply"

    # Test scroll area
    scroll_areas = filter_widget.findChildren(QScrollArea)
    assert len(scroll_areas) == 1
    scroll_area = scroll_areas[0]
    assert scroll_area.widgetResizable()

    # Test initial visibility of filter widgets
    assert filter_widget.median_filter_widget.isHidden()
    assert filter_widget.wavelet_filter_widget.isHidden()


def test_filter_method_switching(make_viewer_model, qtbot):
    """Test switching between median and wavelet filter methods."""
    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    assert filter_widget.filter_method_combobox.currentText() == "None"
    assert filter_widget.median_filter_widget.isHidden()
    assert filter_widget.wavelet_filter_widget.isHidden()

    filter_widget.filter_method_combobox.setCurrentText("Median")
    filter_widget.on_filter_method_changed()

    assert not filter_widget.median_filter_widget.isHidden()
    assert filter_widget.wavelet_filter_widget.isHidden()

    filter_widget.filter_method_combobox.setCurrentText(
        "Wavelet (binlet pawFLIM)"
    )
    filter_widget.on_filter_method_changed()

    assert filter_widget.median_filter_widget.isHidden()
    assert not filter_widget.wavelet_filter_widget.isHidden()

    filter_widget.filter_method_combobox.setCurrentText("Median")
    filter_widget.on_filter_method_changed()

    assert not filter_widget.median_filter_widget.isHidden()
    assert filter_widget.wavelet_filter_widget.isHidden()


def test_threshold_method_none_option(make_viewer_model, qtbot):
    """Test the 'None' threshold method option."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    # Set threshold to some value first
    filter_widget.threshold_slider.setValue((10, 37))
    filter_widget.on_threshold_slider_change()
    assert filter_widget.threshold_slider.value() == (10, 37)

    filter_widget.threshold_method_combobox.setCurrentText("None")
    filter_widget.on_threshold_method_changed()

    lower_val, upper_val = filter_widget.threshold_slider.value()
    assert lower_val == 0
    assert filter_widget.min_threshold_edit.text() == "0.00"


def test_automatic_threshold_methods(make_viewer_model, qtbot):
    """Test automatic threshold calculation methods."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    filter_widget.threshold_method_combobox.setCurrentText("Otsu")
    filter_widget.on_threshold_method_changed()
    otsu_lower, otsu_upper = filter_widget.threshold_slider.value()
    assert otsu_lower > 0

    filter_widget.threshold_method_combobox.setCurrentText("Li")
    filter_widget.on_threshold_method_changed()
    li_lower, li_upper = filter_widget.threshold_slider.value()
    assert li_lower > 0

    filter_widget.threshold_method_combobox.setCurrentText("Yen")
    filter_widget.on_threshold_method_changed()
    yen_lower, yen_upper = filter_widget.threshold_slider.value()
    assert yen_lower > 0

    values = [otsu_lower, li_lower, yen_lower]
    assert len(set(values)) >= 1


def test_manual_threshold_switching(make_viewer_model, qtbot):
    """Test that manually changing slider switches to Manual mode."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    filter_widget.threshold_method_combobox.setCurrentText("Otsu")
    filter_widget.on_threshold_method_changed()
    assert filter_widget.threshold_method_combobox.currentText() == "Otsu"

    filter_widget.threshold_slider.setValue((42, 80))
    filter_widget.on_threshold_slider_change()

    assert filter_widget.threshold_method_combobox.currentText() == "Manual"


def test_none_threshold_slider_behavior(make_viewer_model, qtbot):
    """Test that None threshold doesn't switch to Manual when slider is at full range."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    filter_widget.threshold_method_combobox.setCurrentText("None")
    filter_widget.on_threshold_method_changed()
    assert filter_widget.threshold_method_combobox.currentText() == "None"
    lower_val, upper_val = filter_widget.threshold_slider.value()
    assert lower_val == 0

    filter_widget.on_threshold_slider_change()

    assert filter_widget.threshold_method_combobox.currentText() == "None"


def create_image_layer_with_incompatible_harmonics():
    """Create an image layer with incompatible harmonics for wavelet filtering."""
    layer = create_image_layer_with_phasors()

    # Update harmonics in the new array-based metadata structure
    # Incompatible harmonics are non-consecutive (e.g., [1, 3, 5] instead of [1, 2])
    layer.metadata['harmonics'] = [1, 3, 5]

    return layer


def test_wavelet_harmonics_validation_compatible(make_viewer_model, qtbot):
    """Test wavelet harmonics validation with compatible harmonics."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()

    # Set compatible harmonics (consecutive: 1, 2)
    intensity_image_layer.metadata['harmonics'] = [1, 2]

    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    filter_widget.filter_method_combobox.setCurrentText(
        "Wavelet (binlet pawFLIM)"
    )
    filter_widget.on_filter_method_changed()

    assert filter_widget.harmonic_warning_label.isHidden()
    assert not filter_widget.wavelet_params_widget.isHidden()


def test_wavelet_harmonics_validation_incompatible(make_viewer_model, qtbot):
    """Test wavelet harmonics validation with incompatible harmonics."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_incompatible_harmonics()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    filter_widget.filter_method_combobox.setCurrentText(
        "Wavelet (binlet pawFLIM)"
    )
    filter_widget.on_filter_method_changed()

    assert not filter_widget.harmonic_warning_label.isHidden()
    assert filter_widget.wavelet_params_widget.isHidden()

    warning_text = filter_widget.harmonic_warning_label.text()
    assert "Warning: Harmonics" in warning_text
    assert "not compatible" in warning_text


def test_apply_button_with_wavelet_filter(make_viewer_model, qtbot):
    """Test apply button with wavelet filter method."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()

    # Set compatible harmonics (consecutive: 1, 2)
    intensity_image_layer.metadata['harmonics'] = [1, 2]

    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    filter_widget.filter_method_combobox.setCurrentText(
        "Wavelet (binlet pawFLIM)"
    )
    filter_widget.on_filter_method_changed()
    filter_widget.wavelet_sigma_spinbox.setValue(1.5)
    filter_widget.wavelet_levels_spinbox.setValue(2)
    filter_widget.threshold_slider.setValue((10, 90))
    filter_widget.threshold_method_combobox.setCurrentText("Manual")

    with (
        patch(
            'napari_phasors.filter_tab.apply_filter_and_threshold'
        ) as mock_apply,
        patch.object(parent, 'plot') as mock_plot,
    ):
        filter_widget.apply_button_clicked()

        mock_apply.assert_called_once()
        call_args = mock_apply.call_args

        assert call_args[0][0] == intensity_image_layer
        assert call_args[1]['filter_method'] == 'wavelet'
        assert call_args[1]['sigma'] == 1.5
        assert call_args[1]['levels'] == 2
        assert 'harmonics' in call_args[1]

        mock_plot.assert_called_once()


def test_apply_button_with_median_filter(make_viewer_model, qtbot):
    """Test apply button with median filter method."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    filter_widget.filter_method_combobox.setCurrentText("Median")
    filter_widget.on_filter_method_changed()
    filter_widget.median_filter_spinbox.setValue(5)
    filter_widget.median_filter_repetition_spinbox.setValue(2)
    filter_widget.threshold_slider.setValue((10, 90))
    filter_widget.threshold_method_combobox.setCurrentText("Manual")

    with (
        patch(
            'napari_phasors.filter_tab.apply_filter_and_threshold'
        ) as mock_apply,
        patch.object(parent, 'plot'),
    ):
        filter_widget.apply_button_clicked()

        mock_apply.assert_called_once()
        call_args = mock_apply.call_args

        assert call_args[0][0] == intensity_image_layer
        assert call_args[1]['filter_method'] == 'median'
        assert call_args[1]['size'] == 5
        assert call_args[1]['repeat'] == 2


def test_threshold_method_storage_in_metadata(make_viewer_model, qtbot):
    """Test that threshold method is stored in layer metadata when applying."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    filter_widget.threshold_method_combobox.setCurrentText("Li")

    with patch(
        'napari_phasors.filter_tab.apply_filter_and_threshold'
    ) as mock_apply:
        filter_widget.apply_button_clicked()

        # Check that apply_filter_and_threshold was called with correct parameters
        mock_apply.assert_called_once()
        call_args = mock_apply.call_args

        # Verify threshold_method was passed to the function
        assert call_args[1]['threshold_method'] == "Li"


def test_calculate_automatic_threshold(make_viewer_model, qtbot):
    """Test automatic threshold calculation methods."""
    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    test_data = np.random.rand(100, 100) * 100

    otsu_lower = filter_widget.calculate_automatic_threshold("Otsu", test_data)
    li_lower = filter_widget.calculate_automatic_threshold("Li", test_data)
    yen_lower = filter_widget.calculate_automatic_threshold("Yen", test_data)

    assert isinstance(otsu_lower, (int, float))
    assert isinstance(li_lower, (int, float))
    assert isinstance(yen_lower, (int, float))
    assert otsu_lower >= 0
    assert li_lower >= 0
    assert yen_lower >= 0

    empty_data = np.array([])
    lower = filter_widget.calculate_automatic_threshold("Otsu", empty_data)
    assert lower == 0

    nan_data = np.full((10, 10), np.nan)
    lower = filter_widget.calculate_automatic_threshold("Otsu", nan_data)
    assert lower == 0


def test_settings_restoration_with_wavelet(make_viewer_model, qtbot):
    """Test that wavelet settings are properly restored from metadata."""
    viewer = make_viewer_model()
    harmonics = [1, 2]
    intensity_image_layer = create_image_layer_with_phasors(harmonic=harmonics)

    # Set compatible harmonics (consecutive: 1, 2)
    intensity_image_layer.metadata['harmonics'] = harmonics

    intensity_image_layer.metadata["settings"] = {
        "threshold": 0.1,
        "threshold_upper": 5.0,
        "threshold_method": "Li",
        "filter": {
            "method": "wavelet",
            "sigma": 3.5,
            "levels": 4,
        },
    }

    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    assert (
        filter_widget.filter_method_combobox.currentText()
        == "Wavelet (binlet pawFLIM)"
    )
    assert (
        filter_widget.filter_method_combobox.currentText()
        == "Wavelet (binlet pawFLIM)"
    )
    assert filter_widget.wavelet_sigma_spinbox.value() == 3.5
    assert filter_widget.wavelet_levels_spinbox.value() == 4
    assert filter_widget.threshold_method_combobox.currentText() == "Li"

    # Check that both thresholds are restored
    lower_val, upper_val = filter_widget.threshold_slider.value()
    assert lower_val == int(0.1 * filter_widget.threshold_factor)

    # Upper value should be restored from settings, but clamped to slider maximum
    # Since 5.0 is much larger than the actual max mean value, it will be clamped
    expected_upper = min(
        int(5.0 * filter_widget.threshold_factor),
        filter_widget.threshold_slider.maximum(),
    )
    assert upper_val == expected_upper


def test_settings_restoration_with_incompatible_wavelet(
    make_viewer_model, qtbot
):
    """Test that incompatible wavelet settings fall back to median."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_incompatible_harmonics()

    intensity_image_layer.metadata["settings"] = {
        "filter": {
            "method": "wavelet",
            "sigma": 3.5,
            "levels": 4,
        },
    }

    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    assert filter_widget.filter_method_combobox.currentText() == "Median"


def test_filter_widget_histogram_styling(make_viewer_model, qtbot):
    """Test that histogram styling is applied correctly."""
    import matplotlib.colors as mcolors

    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    # Verify that style_histogram_axes was called during initialization
    assert filter_widget.hist_ax.patch.get_alpha() == 0
    assert filter_widget.hist_fig.patch.get_alpha() == 0

    grey_rgba = mcolors.to_rgba('grey')

    for spine in filter_widget.hist_ax.spines.values():
        np.testing.assert_array_almost_equal(spine.get_edgecolor(), grey_rgba)
        assert spine.get_linewidth() == 1

    assert filter_widget.hist_ax.get_ylabel() == "Count"
    assert filter_widget.hist_ax.get_xlabel() == "Mean Intensity"


def test_filter_widget_with_layer_data(make_viewer_model, qtbot):
    """Test filter widget behavior with actual layer data."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)

    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    max_mean_value = np.nanmax(intensity_image_layer.metadata["original_mean"])
    expected_magnitude = int(np.log10(max_mean_value))
    expected_threshold_factor = (
        10 ** (2 - expected_magnitude) if expected_magnitude <= 2 else 1
    )
    assert filter_widget.threshold_factor == expected_threshold_factor

    expected_max = int(
        np.ceil(max_mean_value * filter_widget.threshold_factor)
    )
    assert filter_widget.threshold_slider.maximum() == expected_max

    # With "None" as default, slider should be at full range
    actual_lower, actual_upper = filter_widget.threshold_slider.value()
    assert actual_lower == 0
    assert actual_upper == expected_max

    assert filter_widget.threshold_method_combobox.currentText() == "None"


def test_filter_widget_threshold_slider_callback(make_viewer_model, qtbot):
    """Test threshold slider callback functionality."""
    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    filter_widget.threshold_factor = 10

    test_lower = 20
    test_upper = 80
    filter_widget.threshold_slider.setValue((test_lower, test_upper))
    filter_widget.on_threshold_slider_change()

    expected_min_text = f"{test_lower / filter_widget.threshold_factor:.2f}"
    expected_max_text = f"{test_upper / filter_widget.threshold_factor:.2f}"
    assert filter_widget.min_threshold_edit.text() == expected_min_text
    assert filter_widget.max_threshold_edit.text() == expected_max_text


def test_filter_widget_kernel_size_callback(make_viewer_model, qtbot):
    """Test kernel size spinbox callback functionality."""
    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    test_value = 5
    filter_widget.median_filter_spinbox.setValue(test_value)
    filter_widget.on_median_kernel_size_change()

    expected_text = f"Kernel Size: {test_value} x {test_value}"
    assert filter_widget.median_filter_label.text() == expected_text


def test_spinbox_and_slider_do_not_call_plot(make_viewer_model, qtbot):
    """Changing spinbox or slider does not call parent.plot() unless apply is clicked."""
    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    with patch.object(parent, 'plot') as mock_plot:
        filter_widget.median_filter_spinbox.setValue(5)
        filter_widget.threshold_slider.setValue((5, 20))

        mock_plot.assert_not_called()


def test_slider_value_modifies_threshold_line(make_viewer_model, qtbot):
    """Changing the slider value updates the threshold line position."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab
    filter_widget._on_image_layer_changed()
    filter_widget.plot_mean_histogram()

    filter_widget.threshold_slider.setValue((5, 20))
    filter_widget.on_threshold_slider_change()
    assert filter_widget.threshold_line_lower is not None
    assert filter_widget.threshold_line_upper is not None
    expected_x_lower = (
        filter_widget.threshold_slider.value()[0]
        / filter_widget.threshold_factor
    )
    expected_x_upper = (
        filter_widget.threshold_slider.value()[1]
        / filter_widget.threshold_factor
    )
    line_data_lower = filter_widget.threshold_line_lower.get_xdata()
    line_data_upper = filter_widget.threshold_line_upper.get_xdata()
    assert (
        line_data_lower[0] == expected_x_lower
        and line_data_lower[1] == expected_x_lower
    )
    assert (
        line_data_upper[0] == expected_x_upper
        and line_data_upper[1] == expected_x_upper
    )


def test_no_plot_called_if_combobox_empty(make_viewer_model, qtbot):
    """If combobox is empty, plot methods are not called."""
    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    with patch.object(parent, 'plot') as mock_plot:
        with patch('napari_phasors.filter_tab.apply_filter_and_threshold'):
            filter_widget.apply_button.click()
        mock_plot.assert_not_called()

    filter_widget.plot_mean_histogram()
    filter_widget.update_threshold_lines()


def test_slider_and_histogram_update_on_layer_add_and_select(
    make_viewer_model,
    qtbot,
):
    """Adding a new layer updates slider range, threshold line, and histogram only when selected."""
    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    assert filter_widget.threshold_slider.maximum() == 100

    intensity_image_layer1 = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer1)

    # Switch to filter tab to trigger histogram plotting
    filter_tab_index = parent.tab_widget.indexOf(filter_widget)
    parent.tab_widget.setCurrentIndex(filter_tab_index)

    expected_max1 = int(
        np.ceil(
            np.nanmax(intensity_image_layer1.metadata["original_mean"])
            * filter_widget.threshold_factor
        )
    )
    assert filter_widget.threshold_slider.maximum() == expected_max1
    assert filter_widget.threshold_line_lower is not None
    assert filter_widget.threshold_line_upper is not None
    assert len(filter_widget.hist_ax.patches) > 0

    intensity_image_layer2 = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer2)

    prev_max = filter_widget.threshold_slider.maximum()
    assert filter_widget.threshold_slider.maximum() == prev_max

    parent.image_layer_with_phasor_features_combobox.setCurrentText(
        intensity_image_layer2.name
    )

    # Ensure filter tab is still active to trigger histogram update
    parent.tab_widget.setCurrentIndex(filter_tab_index)

    expected_max2 = int(
        np.ceil(
            np.nanmax(intensity_image_layer2.metadata["original_mean"])
            * filter_widget.threshold_factor
        )
    )
    assert filter_widget.threshold_slider.maximum() == expected_max2
    assert filter_widget.threshold_line_lower is not None
    assert filter_widget.threshold_line_upper is not None
    assert len(filter_widget.hist_ax.patches) > 0


def test_layer_with_no_phasor_features_does_nothing(make_viewer_model, qtbot):
    """If a layer with no phasor features is added, nothing should happen."""
    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    # Create a dummy napari Image layer without phasor features
    dummy_data = np.random.rand(10, 10)
    image_layer = Image(dummy_data, name="no_phasor_layer")
    viewer.add_layer(image_layer)

    # Add a regular image layer (no phasor features)
    regular_layer = Image(np.random.random((10, 10)))
    viewer.add_layer(regular_layer)

    # The combobox should not be updated (no items checked)
    # since neither layer has phasor features
    assert len(parent.image_layers_checkable_combobox.checkedItems()) == 0

    # Threshold lines should not be created without phasor data
    assert filter_widget.threshold_line_lower is None
    assert filter_widget.threshold_line_upper is None


def test_filter_widget_layer_with_settings(make_viewer_model, qtbot):
    """Test filter widget behavior when layer has existing settings."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()

    intensity_image_layer.metadata["settings"] = {
        "threshold": 0.1,
        "threshold_upper": 5.0,
        "threshold_method": "Li",
        "filter": {"method": "median", "size": 7, "repeat": 3},
    }
    viewer.add_layer(intensity_image_layer)

    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    # Check settings were properly restored
    assert filter_widget.filter_method_combobox.currentText() == "Median"
    lower_val, upper_val = filter_widget.threshold_slider.value()
    assert lower_val == int(0.1 * filter_widget.threshold_factor)

    # Upper value should be restored from settings, but clamped to slider maximum
    expected_upper = min(
        int(5.0 * filter_widget.threshold_factor),
        filter_widget.threshold_slider.maximum(),
    )
    assert upper_val == expected_upper

    assert filter_widget.median_filter_spinbox.value() == 7
    assert filter_widget.median_filter_repetition_spinbox.value() == 3
    assert filter_widget.threshold_method_combobox.currentText() == "Li"


def test_filter_widget_plot_histogram_no_layer(make_viewer_model, qtbot):
    """Test plotting histogram when no layer is available."""
    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    filter_widget.plot_mean_histogram()

    children_before = len(filter_widget.hist_ax.get_children())

    assert children_before >= 0


def test_filter_widget_update_threshold_line_no_layer(
    make_viewer_model, qtbot
):
    """Test updating threshold line when no layer is available."""
    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab
    filter_widget.update_threshold_lines()

    assert filter_widget.threshold_line_lower is None
    assert filter_widget.threshold_line_upper is None


def test_filter_widget_ui_layout(make_viewer_model, qtbot):
    """Test the UI layout structure."""
    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    main_layout = filter_widget.layout()
    assert isinstance(main_layout, QVBoxLayout)

    scroll_areas = filter_widget.findChildren(QScrollArea)
    assert len(scroll_areas) == 1

    # Check that apply button is outside scroll area
    apply_buttons = filter_widget.findChildren(QPushButton)
    apply_button = [btn for btn in apply_buttons if btn.text() == "Apply"][0]
    assert apply_button == filter_widget.apply_button

    h_layouts = filter_widget.findChildren(QHBoxLayout)
    assert len(h_layouts) >= 5


def test_filter_widget_canvas_properties(make_viewer_model, qtbot):
    """Test canvas and figure properties."""
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    assert filter_widget.hist_fig.get_figwidth() == 3.5
    assert filter_widget.hist_fig.get_figheight() == 1.5

    assert filter_widget.hist_fig.get_constrained_layout()

    canvas_widgets = filter_widget.findChildren(FigureCanvasQTAgg)
    assert len(canvas_widgets) == 1

    canvas = canvas_widgets[0]
    assert canvas.height() == 150


def test_log_scale_checkbox_functionality(make_viewer_model, qtbot):
    """Test log scale checkbox functionality."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    filter_widget.plot_mean_histogram()

    assert filter_widget.hist_ax.get_yscale() == 'linear'
    assert not filter_widget.log_scale_checkbox.isChecked()

    filter_widget.log_scale_checkbox.setChecked(True)
    filter_widget.on_log_scale_changed(2)  # Qt.Checked = 2

    assert filter_widget.hist_ax.get_yscale() == 'log'

    filter_widget.log_scale_checkbox.setChecked(False)
    filter_widget.on_log_scale_changed(0)  # Qt.Unchecked = 0

    assert filter_widget.hist_ax.get_yscale() == 'linear'


def test_log_scale_persists_after_new_layer_redraw(make_viewer_model, qtbot):
    """Test that histogram redraws keep the current log scale setting."""
    viewer = make_viewer_model()
    first_layer = create_image_layer_with_phasors()
    viewer.add_layer(first_layer)
    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    filter_tab_index = parent.tab_widget.indexOf(filter_widget)
    parent.tab_widget.setCurrentIndex(filter_tab_index)

    filter_widget.plot_mean_histogram()
    filter_widget.log_scale_checkbox.setChecked(True)
    filter_widget.on_log_scale_changed(2)  # Qt.Checked = 2

    second_layer = create_image_layer_with_phasors()
    viewer.add_layer(second_layer)
    parent.image_layer_with_phasor_features_combobox.setCurrentText(
        second_layer.name
    )
    filter_widget._on_image_layer_changed()

    assert filter_widget.log_scale_checkbox.isChecked()
    assert filter_widget.hist_ax.get_yscale() == 'log'


def test_log_scale_checkbox_with_no_layer(make_viewer_model, qtbot):
    """Test log scale checkbox when no layer is selected."""
    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    # Change log scale state without a layer
    filter_widget.log_scale_checkbox.setChecked(True)
    filter_widget.on_log_scale_changed(True)

    # Should not crash
    assert filter_widget.log_scale_checkbox.isChecked()


def test_filter_widget_uses_masked_region_for_threshold(
    make_viewer_model, qtbot
):
    """Test that filter widget uses masked region when calculating max_mean_value."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    # Add a mask to the layer metadata
    mask_data = np.zeros((2, 5), dtype=int)
    mask_data[:1, :] = 1  # Top half is masked in
    intensity_image_layer.metadata['mask'] = mask_data

    # Trigger layer change to update histogram
    filter_widget._on_image_layer_changed()

    # Verify that threshold slider was configured (using masked region)
    assert filter_widget.threshold_slider.maximum() > 0


def test_filter_widget_without_mask_uses_full_image(make_viewer_model, qtbot):
    """Test filter widget uses full image when no mask is present."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    # Ensure no mask in metadata
    if 'mask' in intensity_image_layer.metadata:
        del intensity_image_layer.metadata['mask']

    # Trigger layer change
    filter_widget._on_image_layer_changed()

    # Should use full image for calculations
    assert filter_widget.threshold_slider.maximum() > 0


def test_min_threshold_edit_changes_slider(make_viewer_model, qtbot):
    """Test that editing min threshold text field updates the slider."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    # First set slider to known values with room to move
    filter_widget.threshold_slider.setValue((10, 80))
    filter_widget.on_threshold_slider_change()

    # Set a new minimum threshold value that's between current min and max
    new_min_value = (
        0.15  # Choose a value that will be between 10 and 80 when scaled
    )
    filter_widget.min_threshold_edit.setText(str(new_min_value))
    filter_widget.on_min_threshold_edit_changed()

    # Check that slider was updated
    lower_val, _ = filter_widget.threshold_slider.value()
    expected_slider_value = int(new_min_value * filter_widget.threshold_factor)
    assert lower_val == expected_slider_value

    # Check that threshold method switched to Manual
    assert filter_widget.threshold_method_combobox.currentText() == "Manual"


def test_max_threshold_edit_changes_slider(make_viewer_model, qtbot):
    """Test that editing max threshold text field updates the slider."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    # Set a new maximum threshold value
    max_mean = np.nanmax(intensity_image_layer.metadata["original_mean"])
    new_max_value = max_mean * 0.8
    filter_widget.max_threshold_edit.setText(f"{new_max_value:.2f}")
    filter_widget.on_max_threshold_edit_changed()

    # Check that slider was updated
    _, upper_val = filter_widget.threshold_slider.value()
    # The value gets rounded when converted to text with .2f format, then converted back
    # So we need to calculate the expected value the same way
    text_value = float(f"{new_max_value:.2f}")
    expected_slider_value = int(text_value * filter_widget.threshold_factor)
    assert upper_val == expected_slider_value

    # Check that threshold method switched to Manual
    assert filter_widget.threshold_method_combobox.currentText() == "Manual"


def test_min_threshold_edit_clamped_to_max(make_viewer_model, qtbot):
    """Test that min threshold cannot exceed max threshold."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    # Set slider to known values
    filter_widget.threshold_slider.setValue((10, 50))
    filter_widget.on_threshold_slider_change()

    # Try to set min higher than max
    filter_widget.min_threshold_edit.setText("100.0")
    filter_widget.on_min_threshold_edit_changed()

    # Min should be clamped to max
    lower_val, upper_val = filter_widget.threshold_slider.value()
    assert lower_val <= upper_val


def test_max_threshold_edit_clamped_to_min(make_viewer_model, qtbot):
    """Test that max threshold cannot go below min threshold."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    # Set slider to known values
    filter_widget.threshold_slider.setValue((30, 80))
    filter_widget.on_threshold_slider_change()

    # Try to set max lower than min
    filter_widget.max_threshold_edit.setText("0.5")
    filter_widget.on_max_threshold_edit_changed()

    # Max should be clamped to min
    lower_val, upper_val = filter_widget.threshold_slider.value()
    assert upper_val >= lower_val


def test_invalid_threshold_edit_resets_to_current(make_viewer_model, qtbot):
    """Test that invalid input in threshold edits resets to current value."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    # Get current slider values
    lower_val, upper_val = filter_widget.threshold_slider.value()
    expected_min = f"{lower_val / filter_widget.threshold_factor:.2f}"
    expected_max = f"{upper_val / filter_widget.threshold_factor:.2f}"

    # Try invalid inputs
    filter_widget.min_threshold_edit.setText("invalid")
    filter_widget.on_min_threshold_edit_changed()
    assert filter_widget.min_threshold_edit.text() == expected_min

    filter_widget.max_threshold_edit.setText("also_invalid")
    filter_widget.on_max_threshold_edit_changed()
    assert filter_widget.max_threshold_edit.text() == expected_max


def test_threshold_edit_updates_histogram_lines(make_viewer_model, qtbot):
    """Test that editing threshold values updates the histogram lines."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab
    filter_widget.plot_mean_histogram()

    # Set slider to known values first
    filter_widget.threshold_slider.setValue((10, 80))
    filter_widget.on_threshold_slider_change()

    # Edit min threshold to a value within the valid range
    new_min_value = 0.2
    filter_widget.min_threshold_edit.setText(str(new_min_value))
    filter_widget.on_min_threshold_edit_changed()

    # Check that threshold line exists and is at correct position
    assert filter_widget.threshold_line_lower is not None
    expected_x = (
        int(new_min_value * filter_widget.threshold_factor)
        / filter_widget.threshold_factor
    )
    line_data = filter_widget.threshold_line_lower.get_xdata()
    assert abs(line_data[0] - expected_x) < 0.01


def test_filter_widget_no_duplicate_signal_connections(
    make_viewer_model, qtbot
):
    """Test that switching tabs does not accumulate signal connections.

    Regression test: _update_histogram_if_needed() previously connected
    signals every time it was called (on each tab switch), causing
    callbacks to fire N times after N tab switches.
    """
    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    def _receiver_count(combo_box):
        signal = combo_box.currentTextChanged
        with contextlib.suppress(TypeError):
            return combo_box.receivers(signal)

        # PySide variants expect a signal signature string.
        for signal_name in (
            "currentTextChanged(str)",
            "currentTextChanged(QString)",
        ):
            with contextlib.suppress(TypeError):
                return combo_box.receivers(signal_name)

        raise AssertionError(
            "Could not query signal receivers for currentTextChanged"
        )

    # Count receivers on the threshold_method_combobox signal before
    initial_receivers = _receiver_count(
        filter_widget.threshold_method_combobox
    )

    # Simulate switching to the filter tab multiple times
    for _ in range(5):
        filter_widget._update_histogram_if_needed()

    # Count receivers after — should be the same as before
    final_receivers = _receiver_count(filter_widget.threshold_method_combobox)

    assert final_receivers == initial_receivers, (
        f"Signal has {final_receivers} receivers after 5 tab switches "
        f"(started with {initial_receivers}). "
        f"Connections are accumulating on each call to "
        f"_update_histogram_if_needed()."
    )


def test_apply_threshold_invalidates_features_cache(make_viewer_model, qtbot):
    """Regression test for applying threshold must update the phasor plot.

    The plotter caches merged features keyed on (selected layer names,
    harmonic). After applying a filter/threshold, the layer's G/S arrays
    change but the cache key stays the same, so the cached stale features
    must be invalidated by ``refresh_phasor_data``.
    """
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)

    # Prime the features cache by getting features once.
    initial_features = parent.get_merged_features()
    assert initial_features is not None
    assert parent._features_cache is not None

    # Inject a sentinel into the cache that we can detect later. Use a
    # plain Python tuple of strings so equality/identity checks are
    # unambiguous (numpy arrays would raise on ==).
    sentinel = ('stale-sentinel', 'data')
    parent._features_cache = sentinel

    # Calling refresh_phasor_data() must invalidate the cache so the
    # next call to get_merged_features() recomputes from layer metadata.
    parent.refresh_phasor_data()

    assert parent._features_cache is not sentinel, (
        "refresh_phasor_data() did not invalidate stale features cache "
        "— threshold/filter changes will not be reflected in the plot."
    )
    # The cache should either be None or repopulated with fresh data
    # (not the sentinel and not the same object as before injection).
    assert parent._features_cache is None or (
        id(parent._features_cache) != id(sentinel)
    )


def test_apply_threshold_marks_deferred_tabs_for_update(
    make_viewer_model, qtbot
):
    """Regression test deferred tabs must be marked stale after
    a filter/threshold is applied so they refresh when next visible.
    """
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    # Ensure the deferrable tabs start in a clean state.
    for tab_attr in ('phasor_mapping_tab', 'components_tab', 'fret_tab'):
        if hasattr(parent, tab_attr):
            getattr(parent, tab_attr)._needs_update = False

    # Switch to a non-deferrable tab so deferred tabs don't auto-restore.
    parent.tab_widget.setCurrentWidget(filter_widget)

    filter_widget.threshold_method_combobox.setCurrentText("Manual")
    filter_widget.threshold_slider.setValue((20, 90))
    filter_widget.apply_button_clicked()

    for tab_attr in ('phasor_mapping_tab', 'components_tab', 'fret_tab'):
        if hasattr(parent, tab_attr):
            tab = getattr(parent, tab_attr)
            assert tab._needs_update is True, (
                f"{tab_attr} was not marked for deferred update after "
                f"filter/threshold was applied."
            )


def test_filter_threshold_line_mouse_drag(make_viewer_model, qtbot):
    """Cover the threshold-line press/move/release mouse handlers."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)
    parent = PlotterWidget(viewer)
    fw = parent.filter_tab
    fw._on_image_layer_changed()
    fw.plot_mean_histogram()
    fw.threshold_slider.setValue((5, 20))
    fw.on_threshold_slider_change()
    factor = fw.threshold_factor

    class Ev:
        def __init__(self, x, inaxes):
            self.xdata = x
            self.ydata = 0.0
            self.inaxes = inaxes
            self.button = 1

    lower_x = 5 / factor
    upper_x = 20 / factor

    # Press on the lower line, drag it, release.
    fw.on_mouse_press(Ev(lower_x, fw.hist_ax))
    assert fw._dragging_line == "lower"
    fw.on_mouse_move(Ev(10 / factor, fw.hist_ax))
    fw.on_mouse_release(Ev(10 / factor, fw.hist_ax))
    assert fw._dragging_line is None

    # Press on the upper line, drag it, release.
    fw.on_mouse_press(Ev(upper_x, fw.hist_ax))
    assert fw._dragging_line == "upper"
    fw.on_mouse_move(Ev(15 / factor, fw.hist_ax))
    fw.on_mouse_release(Ev(15 / factor, fw.hist_ax))
    assert fw._dragging_line is None

    # Press outside the axes does nothing.
    fw.on_mouse_press(Ev(lower_x, None))
    assert fw._dragging_line is None
    # Move while not dragging is a no-op.
    fw.on_mouse_move(Ev(lower_x, fw.hist_ax))


def test_filter_on_image_layer_changed_restores_full_settings(
    make_viewer_model, qtbot
):
    """A layer carrying full filter+threshold settings restores all widgets."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    layer.metadata["settings"] = {
        "filter": {
            "method": "wavelet",
            "size": 5,
            "repeat": 2,
            "sigma": 3.0,
            "levels": 4,
        },
        "threshold": 2.0,
        "threshold_upper": 8.0,
    }
    viewer.add_layer(layer)
    parent = PlotterWidget(viewer)
    fw = parent.filter_tab
    fw._on_image_layer_changed()
    assert fw.median_filter_spinbox.value() == 5
    assert fw.median_filter_repetition_spinbox.value() == 2
    assert fw.wavelet_sigma_spinbox.value() == 3.0
    assert fw.wavelet_levels_spinbox.value() == 4


def test_slider_minimum_tracks_data_minimum(make_viewer_model, qtbot):
    """An image whose intensities start above zero bounds the slider minimum.

    Regression: importing an already-thresholded image left the lower
    threshold line to the left of the visible histogram because the slider
    minimum was hard-coded to zero.
    """
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    # Simulate an already-thresholded intensity image: nothing below 5.0.
    mean = layer.metadata["original_mean"].astype(float)
    mean[:] = np.linspace(5.0, 40.0, mean.size).reshape(mean.shape)
    layer.metadata["original_mean"] = mean
    layer.metadata["settings"] = {}  # no stored threshold
    viewer.add_layer(layer)
    parent = PlotterWidget(viewer)
    fw = parent.filter_tab
    fw._on_image_layer_changed()

    expected_min = int(5.0 * fw.threshold_factor)
    assert fw.threshold_slider.minimum() == expected_min
    lower_val, _ = fw.threshold_slider.value()
    assert lower_val == expected_min
    assert fw.min_threshold_edit.text() == f"{5.0:.2f}"


def test_apply_stores_none_for_unconstrained_bounds(make_viewer_model, qtbot):
    """Threshold handles left at the slider extremes persist as ``None``.

    Regression: leaving the max handle at the top stored the current data max
    as an explicit ``threshold_upper``. While a mask was active this froze the
    reduced max, which then persisted after the mask was removed.
    """
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)
    parent = PlotterWidget(viewer)
    fw = parent.filter_tab

    # Manual thresholding with both handles left at their extremes.
    fw.threshold_slider.setValue(
        (fw.threshold_slider.minimum(), fw.threshold_slider.maximum())
    )
    fw.threshold_method_combobox.setCurrentText("Manual")

    with patch(
        'napari_phasors.filter_tab.apply_filter_and_threshold'
    ) as mock_apply:
        fw.apply_button_clicked()
        call_kwargs = mock_apply.call_args[1]
        assert call_kwargs['threshold'] is None
        assert call_kwargs['threshold_upper'] is None

    # A constrained max should still be persisted as a concrete value.
    upper = fw.threshold_slider.maximum() - 1
    fw.threshold_slider.setValue((fw.threshold_slider.minimum(), upper))
    with patch(
        'napari_phasors.filter_tab.apply_filter_and_threshold'
    ) as mock_apply:
        fw.apply_button_clicked()
        call_kwargs = mock_apply.call_args[1]
        assert call_kwargs['threshold_upper'] == upper / fw.threshold_factor


def test_unconstrained_upper_sits_at_slider_maximum(make_viewer_model, qtbot):
    """A ``None`` stored upper threshold pins the handle to the slider maximum.

    Regression: because the slider maximum is ``ceil``-rounded while the upper
    value was ``int``-truncated, an unconstrained max landed one tick below the
    maximum, was mistaken for a user constraint, and got frozen.
    """
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    layer.metadata["settings"] = {
        "threshold": 0.05,
        "threshold_upper": None,
        "threshold_method": "Manual",
    }
    viewer.add_layer(layer)
    parent = PlotterWidget(viewer)
    fw = parent.filter_tab
    fw._on_image_layer_changed()

    _, upper_val = fw.threshold_slider.value()
    assert upper_val == fw.threshold_slider.maximum()


def test_max_threshold_restored_after_mask_removed(make_viewer_model, qtbot):
    """Removing a mask restores the unconstrained max to the full-data max.

    Regression: masking lowered the slider maximum, and the reduced max stuck
    after the mask was removed instead of returning to the original range.
    """
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    layer.metadata["settings"] = {
        "threshold": 0.05,
        "threshold_method": "Manual",
    }
    viewer.add_layer(layer)
    parent = PlotterWidget(viewer)
    fw = parent.filter_tab
    fw._on_image_layer_changed()

    om = layer.metadata["original_mean"]
    full_max_real = np.nanmax(om)
    full_factor = fw.threshold_factor
    full_max = fw.threshold_slider.maximum()
    _, upper_full = fw.threshold_slider.value()
    # Unconstrained upper pinned to the maximum (real units ~ data max).
    assert upper_full == full_max

    # Mask in only the below-median pixels so the visible max drops.
    masked_max_real = np.nanmax(om[om <= np.median(om)])
    layer.metadata["mask"] = (om <= np.median(om)).astype(int)
    fw._on_image_layer_changed()

    # The visible max drops in real units (the scaled value is not comparable
    # because ``threshold_factor`` rescales with the magnitude).
    assert masked_max_real < full_max_real
    _, upper_masked = fw.threshold_slider.value()
    assert upper_masked == fw.threshold_slider.maximum()
    assert upper_masked / fw.threshold_factor < full_max_real

    # Remove the mask: the full range and unconstrained max must return.
    del layer.metadata["mask"]
    fw._on_image_layer_changed()

    assert fw.threshold_factor == full_factor
    assert fw.threshold_slider.maximum() == full_max
    _, upper_restored = fw.threshold_slider.value()
    assert upper_restored == full_max


def test_histogram_cleared_when_no_layer_selected(make_viewer_model, qtbot):
    """Deselecting all phasor layers clears the intensity histogram."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)
    parent = PlotterWidget(viewer)
    fw = parent.filter_tab
    fw._on_image_layer_changed()
    fw.plot_mean_histogram()
    assert fw._histogram_data is not None
    assert len(fw.hist_ax.patches) > 0

    # Simulate all layers deselected in the phasor-layer combobox.
    with patch.object(parent, 'get_selected_layers', return_value=[]):
        fw._on_image_layer_changed()

    assert fw._histogram_data is None
    assert len(fw.hist_ax.patches) == 0
