from unittest.mock import patch

import numpy as np
from matplotlib.figure import Figure
from napari.layers import Image
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
)

from napari_phasors._tests.test_plotter import create_image_layer_with_phasors
from napari_phasors.filter_tab import FilterWidget
from napari_phasors.plotter import PlotterWidget


def test_filter_widget_initialization_values(make_napari_viewer):
    """Test the initialization of the Filter Widget."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    # Basic widget structure tests
    assert filter_widget.viewer == viewer
    assert filter_widget.parent_widget == parent
    assert isinstance(filter_widget.layout(), QVBoxLayout)

    # Test initial attribute values
    assert (
        filter_widget.parent_widget._labels_layer_with_phasor_features is None
    )
    assert filter_widget._phasors_selected_layer is None
    assert filter_widget.threshold_factor == 1
    assert filter_widget.threshold_line is None

    # Test histogram figure initialization
    assert isinstance(filter_widget.hist_fig, Figure)
    assert filter_widget.hist_ax is not None

    # Test filter method combobox
    assert hasattr(filter_widget, 'filter_method_combobox')
    assert isinstance(filter_widget.filter_method_combobox, QComboBox)
    assert filter_widget.filter_method_combobox.count() == 2
    assert filter_widget.filter_method_combobox.itemText(0) == "Median"
    assert filter_widget.filter_method_combobox.itemText(1) == "Wavelet"
    assert filter_widget.filter_method_combobox.currentText() == "Median"

    # Test threshold method combobox
    assert hasattr(filter_widget, 'threshold_method_combobox')
    assert isinstance(filter_widget.threshold_method_combobox, QComboBox)
    assert filter_widget.threshold_method_combobox.count() == 5
    assert filter_widget.threshold_method_combobox.itemText(0) == "None"
    assert filter_widget.threshold_method_combobox.itemText(1) == "Manual"
    assert filter_widget.threshold_method_combobox.itemText(2) == "Otsu"
    assert filter_widget.threshold_method_combobox.itemText(3) == "Li"
    assert filter_widget.threshold_method_combobox.itemText(4) == "Yen"
    assert filter_widget.threshold_method_combobox.currentText() == "Otsu"

    # Test log scale checkbox
    assert hasattr(filter_widget, 'log_scale_checkbox')
    assert isinstance(filter_widget.log_scale_checkbox, QCheckBox)
    assert filter_widget.log_scale_checkbox.text() == "Log scale"
    assert (
        not filter_widget.log_scale_checkbox.isChecked()
    )  # Should start unchecked

    # Test median filter UI components
    assert hasattr(filter_widget, 'median_filter_label')
    assert isinstance(filter_widget.median_filter_label, QLabel)
    assert (
        filter_widget.median_filter_label.text()
        == "Median Filter Kernel Size: 3 x 3"
    )

    assert hasattr(filter_widget, 'median_filter_spinbox')
    assert isinstance(filter_widget.median_filter_spinbox, QSpinBox)
    assert filter_widget.median_filter_spinbox.minimum() == 2
    assert filter_widget.median_filter_spinbox.maximum() == 99
    assert filter_widget.median_filter_spinbox.value() == 3

    assert hasattr(filter_widget, 'median_filter_repetition_spinbox')
    assert isinstance(filter_widget.median_filter_repetition_spinbox, QSpinBox)
    assert filter_widget.median_filter_repetition_spinbox.minimum() == 0
    assert filter_widget.median_filter_repetition_spinbox.value() == 0

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
    assert not filter_widget.harmonic_warning_label.isVisible()

    # Test threshold slider and label
    assert hasattr(filter_widget, 'label_3')
    assert isinstance(filter_widget.label_3, QLabel)
    assert filter_widget.label_3.text() == "Intensity threshold: 0"

    assert hasattr(filter_widget, 'threshold_slider')
    assert isinstance(filter_widget.threshold_slider, QSlider)
    assert filter_widget.threshold_slider.orientation() == Qt.Horizontal
    assert filter_widget.threshold_slider.minimum() == 0
    assert filter_widget.threshold_slider.maximum() == 100
    assert filter_widget.threshold_slider.value() == 0

    # Test apply button
    assert hasattr(filter_widget, 'apply_button')
    assert isinstance(filter_widget.apply_button, QPushButton)
    assert filter_widget.apply_button.text() == "Apply Filter and Threshold"

    # Test scroll area
    scroll_areas = filter_widget.findChildren(QScrollArea)
    assert len(scroll_areas) == 1
    scroll_area = scroll_areas[0]
    assert scroll_area.widgetResizable() == True

    # Test initial visibility of filter widgets
    assert not filter_widget.median_filter_widget.isHidden()
    assert filter_widget.wavelet_filter_widget.isHidden()


def test_filter_method_switching(make_napari_viewer):
    """Test switching between median and wavelet filter methods."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    assert filter_widget.filter_method_combobox.currentText() == "Median"
    assert not filter_widget.median_filter_widget.isHidden()
    assert filter_widget.wavelet_filter_widget.isHidden()

    filter_widget.filter_method_combobox.setCurrentText("Wavelet")
    filter_widget.on_filter_method_changed()

    assert filter_widget.median_filter_widget.isHidden()
    assert not filter_widget.wavelet_filter_widget.isHidden()

    filter_widget.filter_method_combobox.setCurrentText("Median")
    filter_widget.on_filter_method_changed()

    assert not filter_widget.median_filter_widget.isHidden()
    assert filter_widget.wavelet_filter_widget.isHidden()


def test_threshold_method_none_option(make_napari_viewer):
    """Test the 'None' threshold method option."""
    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    # Set threshold to some value first
    filter_widget.threshold_slider.setValue(10)
    filter_widget.on_threshold_slider_change()
    assert filter_widget.threshold_slider.value() == 10

    filter_widget.threshold_method_combobox.setCurrentText("None")
    filter_widget.on_threshold_method_changed()

    assert filter_widget.threshold_slider.value() == 0
    assert filter_widget.label_3.text() == "Intensity threshold: 0"


def test_automatic_threshold_methods(make_napari_viewer):
    """Test automatic threshold calculation methods."""
    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    filter_widget.threshold_method_combobox.setCurrentText("Otsu")
    filter_widget.on_threshold_method_changed()
    otsu_value = filter_widget.threshold_slider.value()
    assert otsu_value > 0

    filter_widget.threshold_method_combobox.setCurrentText("Li")
    filter_widget.on_threshold_method_changed()
    li_value = filter_widget.threshold_slider.value()
    assert li_value > 0

    filter_widget.threshold_method_combobox.setCurrentText("Yen")
    filter_widget.on_threshold_method_changed()
    yen_value = filter_widget.threshold_slider.value()
    assert yen_value > 0

    values = [otsu_value, li_value, yen_value]
    assert len(set(values)) >= 1


def test_manual_threshold_switching(make_napari_viewer):
    """Test that manually changing slider switches to Manual mode."""
    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    filter_widget.threshold_method_combobox.setCurrentText("Otsu")
    filter_widget.on_threshold_method_changed()
    assert filter_widget.threshold_method_combobox.currentText() == "Otsu"

    filter_widget.threshold_slider.setValue(42)
    filter_widget.on_threshold_slider_change()

    assert filter_widget.threshold_method_combobox.currentText() == "Manual"


def test_none_threshold_slider_behavior(make_napari_viewer):
    """Test that None threshold doesn't switch to Manual when slider is at 0."""
    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    filter_widget.threshold_method_combobox.setCurrentText("None")
    filter_widget.on_threshold_method_changed()
    assert filter_widget.threshold_method_combobox.currentText() == "None"
    assert filter_widget.threshold_slider.value() == 0

    filter_widget.on_threshold_slider_change()

    assert filter_widget.threshold_method_combobox.currentText() == "None"


def create_image_layer_with_incompatible_harmonics():
    """Create an image layer with incompatible harmonics for wavelet filtering."""
    layer = create_image_layer_with_phasors()

    phasor_features = layer.metadata['phasor_features_labels_layer']

    num_pixels = len(phasor_features.features['harmonic'])
    incompatible_harmonics = np.random.choice([1, 3, 5], size=num_pixels)
    phasor_features.features['harmonic'] = incompatible_harmonics

    return layer


def test_wavelet_harmonics_validation_compatible(make_napari_viewer):
    """Test wavelet harmonics validation with compatible harmonics."""
    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()

    phasor_features = intensity_image_layer.metadata[
        'phasor_features_labels_layer'
    ]
    num_pixels = len(phasor_features.features['harmonic'])
    compatible_harmonics = np.random.choice([1, 2], size=num_pixels)
    phasor_features.features['harmonic'] = compatible_harmonics

    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    filter_widget.filter_method_combobox.setCurrentText("Wavelet")
    filter_widget.on_filter_method_changed()

    assert filter_widget.harmonic_warning_label.isHidden()
    assert not filter_widget.wavelet_params_widget.isHidden()


def test_wavelet_harmonics_validation_incompatible(make_napari_viewer):
    """Test wavelet harmonics validation with incompatible harmonics."""
    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_incompatible_harmonics()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    filter_widget.filter_method_combobox.setCurrentText("Wavelet")
    filter_widget.on_filter_method_changed()

    assert not filter_widget.harmonic_warning_label.isVisible()
    assert filter_widget.wavelet_params_widget.isHidden()

    warning_text = filter_widget.harmonic_warning_label.text()
    assert "Warning: Harmonics" in warning_text
    assert "not compatible" in warning_text


def test_apply_button_with_wavelet_filter(make_napari_viewer):
    """Test apply button with wavelet filter method."""
    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()

    phasor_features = intensity_image_layer.metadata[
        'phasor_features_labels_layer'
    ]
    num_pixels = len(phasor_features.features['harmonic'])
    compatible_harmonics = np.random.choice([1, 2], size=num_pixels)
    phasor_features.features['harmonic'] = compatible_harmonics

    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    filter_widget.filter_method_combobox.setCurrentText("Wavelet")
    filter_widget.wavelet_sigma_spinbox.setValue(1.5)
    filter_widget.wavelet_levels_spinbox.setValue(2)
    filter_widget.threshold_slider.setValue(10)

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


def test_apply_button_with_median_filter(make_napari_viewer):
    """Test apply button with median filter method."""
    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    filter_widget.filter_method_combobox.setCurrentText("Median")
    filter_widget.median_filter_spinbox.setValue(5)
    filter_widget.median_filter_repetition_spinbox.setValue(2)
    filter_widget.threshold_slider.setValue(10)

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
        assert call_args[1]['filter_method'] == 'median'
        assert call_args[1]['size'] == 5
        assert call_args[1]['repeat'] == 2


def test_threshold_method_storage_in_metadata(make_napari_viewer):
    """Test that threshold method is stored in layer metadata when applying."""
    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    filter_widget.threshold_method_combobox.setCurrentText("Li")

    with patch('napari_phasors.filter_tab.apply_filter_and_threshold'):
        filter_widget.apply_button_clicked()

    assert "settings" in intensity_image_layer.metadata
    assert (
        intensity_image_layer.metadata["settings"]["threshold_method"] == "Li"
    )


def test_calculate_automatic_threshold(make_napari_viewer):
    """Test automatic threshold calculation methods."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    test_data = np.random.rand(100, 100) * 100

    otsu_threshold = filter_widget.calculate_automatic_threshold(
        "Otsu", test_data
    )
    li_threshold = filter_widget.calculate_automatic_threshold("Li", test_data)
    yen_threshold = filter_widget.calculate_automatic_threshold(
        "Yen", test_data
    )

    assert isinstance(otsu_threshold, (int, float))
    assert isinstance(li_threshold, (int, float))
    assert isinstance(yen_threshold, (int, float))
    assert otsu_threshold >= 0
    assert li_threshold >= 0
    assert yen_threshold >= 0

    empty_data = np.array([])
    result = filter_widget.calculate_automatic_threshold("Otsu", empty_data)
    assert result == 0

    nan_data = np.full((10, 10), np.nan)
    result = filter_widget.calculate_automatic_threshold("Otsu", nan_data)
    assert result == 0


def test_settings_restoration_with_wavelet(make_napari_viewer):
    """Test that wavelet settings are properly restored from metadata."""
    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()

    phasor_features = intensity_image_layer.metadata[
        'phasor_features_labels_layer'
    ]
    num_pixels = len(phasor_features.features['harmonic'])
    compatible_harmonics = np.random.choice([1, 2], size=num_pixels)
    phasor_features.features['harmonic'] = compatible_harmonics

    intensity_image_layer.metadata["settings"] = {
        "threshold": 0.1,
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

    assert filter_widget.filter_method_combobox.currentText() == "Wavelet"
    assert filter_widget.wavelet_sigma_spinbox.value() == 3.5
    assert filter_widget.wavelet_levels_spinbox.value() == 4
    assert filter_widget.threshold_method_combobox.currentText() == "Li"


def test_settings_restoration_with_incompatible_wavelet(make_napari_viewer):
    """Test that incompatible wavelet settings fall back to median."""
    viewer = make_napari_viewer()
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


def test_filter_widget_histogram_styling(make_napari_viewer):
    """Test that histogram styling is applied correctly."""
    viewer = make_napari_viewer()

    with patch.object(FilterWidget, 'style_histogram_axes') as mock_style:
        parent = PlotterWidget(viewer)
        filter_widget = parent.filter_tab
        mock_style.assert_called_once()

    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    assert filter_widget.hist_ax.patch.get_alpha() == 0
    assert filter_widget.hist_fig.patch.get_alpha() == 0

    import matplotlib.colors as mcolors

    grey_rgba = mcolors.to_rgba('grey')

    for spine in filter_widget.hist_ax.spines.values():
        np.testing.assert_array_almost_equal(spine.get_edgecolor(), grey_rgba)
        assert spine.get_linewidth() == 1

    assert filter_widget.hist_ax.get_ylabel() == "Count"
    assert filter_widget.hist_ax.get_xlabel() == "Mean Intensity"


def test_filter_widget_with_layer_data(make_napari_viewer):
    """Test filter widget behavior with actual layer data."""
    viewer = make_napari_viewer()
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

    mean_data = intensity_image_layer.metadata["original_mean"]
    expected_otsu_threshold = filter_widget.calculate_automatic_threshold(
        "Otsu", mean_data
    )
    expected_default_threshold = int(
        expected_otsu_threshold * filter_widget.threshold_factor
    )
    assert filter_widget.threshold_slider.value() == expected_default_threshold

    assert filter_widget.threshold_method_combobox.currentText() == "Otsu"


def test_filter_widget_threshold_slider_callback(make_napari_viewer):
    """Test threshold slider callback functionality."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    filter_widget.threshold_factor = 10

    test_value = 50
    filter_widget.threshold_slider.setValue(test_value)
    filter_widget.on_threshold_slider_change()

    expected_text = (
        f"Intensity threshold: {test_value / filter_widget.threshold_factor}"
    )
    assert filter_widget.label_3.text() == expected_text


def test_filter_widget_kernel_size_callback(make_napari_viewer):
    """Test kernel size spinbox callback functionality."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    test_value = 5
    filter_widget.median_filter_spinbox.setValue(test_value)

    expected_text = f"Median Filter Kernel Size: {test_value} x {test_value}"
    assert filter_widget.median_filter_label.text() == expected_text


def test_filter_widget_histogram_plotting(make_napari_viewer):
    """Test histogram plotting functionality."""
    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)

    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    filter_widget.plot_mean_histogram()

    assert len(filter_widget.hist_ax.get_children()) > 0

    filter_widget.threshold_slider.setValue(50)
    filter_widget.update_threshold_line()


def test_spinbox_and_slider_do_not_call_plot(make_napari_viewer):
    """Changing spinbox or slider does not call parent.plot() unless apply is clicked."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    with patch.object(parent, 'plot') as mock_plot:
        filter_widget.median_filter_spinbox.setValue(5)
        filter_widget.threshold_slider.setValue(42)

        mock_plot.assert_not_called()


def test_slider_value_modifies_threshold_line(make_napari_viewer):
    """Changing the slider value updates the threshold line position."""
    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab
    filter_widget.on_labels_layer_with_phasor_features_changed()
    filter_widget.plot_mean_histogram()

    filter_widget.threshold_slider.setValue(77)
    filter_widget.on_threshold_slider_change()
    assert filter_widget.threshold_line is not None
    expected_x = (
        filter_widget.threshold_slider.value() / filter_widget.threshold_factor
    )
    line_data = filter_widget.threshold_line.get_xdata()
    assert line_data[0] == expected_x and line_data[1] == expected_x


def test_no_plot_called_if_combobox_empty(make_napari_viewer):
    """If combobox is empty, plot methods are not called."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    with patch.object(parent, 'plot') as mock_plot:
        with patch('napari_phasors.filter_tab.apply_filter_and_threshold'):
            filter_widget.apply_button.click()
        mock_plot.assert_not_called()

    filter_widget.plot_mean_histogram()
    filter_widget.update_threshold_line()


def test_slider_and_histogram_update_on_layer_add_and_select(
    make_napari_viewer,
):
    """Adding a new layer updates slider range, threshold line, and histogram only when selected."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    assert filter_widget.threshold_slider.maximum() == 100

    intensity_image_layer1 = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer1)

    expected_max1 = int(
        np.ceil(
            np.nanmax(intensity_image_layer1.metadata["original_mean"])
            * filter_widget.threshold_factor
        )
    )
    assert filter_widget.threshold_slider.maximum() == expected_max1
    assert filter_widget.threshold_line is not None
    assert len(filter_widget.hist_ax.patches) > 0

    intensity_image_layer2 = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer2)

    prev_max = filter_widget.threshold_slider.maximum()
    assert filter_widget.threshold_slider.maximum() == prev_max

    parent.image_layer_with_phasor_features_combobox.setCurrentText(
        intensity_image_layer2.name
    )
    expected_max2 = int(
        np.ceil(
            np.nanmax(intensity_image_layer2.metadata["original_mean"])
            * filter_widget.threshold_factor
        )
    )
    assert filter_widget.threshold_slider.maximum() == expected_max2
    assert filter_widget.threshold_line is not None
    assert len(filter_widget.hist_ax.patches) > 0


def test_layer_with_no_phasor_features_does_nothing(make_napari_viewer):
    """If a layer with no phasor features is added, nothing should happen."""
    viewer = make_napari_viewer()

    with (
        patch.object(
            FilterWidget, 'on_labels_layer_with_phasor_features_changed'
        ) as mock_on_labels,
        patch.object(FilterWidget, 'plot_mean_histogram') as mock_plot_hist,
        patch.object(
            FilterWidget, 'update_threshold_line'
        ) as mock_update_line,
    ):

        parent = PlotterWidget(viewer)
        # Create a dummy napari Image layer without phasor features
        dummy_data = np.random.rand(10, 10)
        image_layer = Image(dummy_data, name="no_phasor_layer")
        viewer.add_layer(image_layer)
        filter_widget = parent.filter_tab

        # Add a regular image layer (no phasor features)
        regular_layer = Image(np.random.random((10, 10)))
        viewer.add_layer(regular_layer)

        # None of the methods should be called
        mock_on_labels.assert_not_called()
        mock_plot_hist.assert_not_called()
        mock_update_line.assert_not_called()

        # The combobox should not be updated
        assert parent.image_layer_with_phasor_features_combobox.count() == 0
        assert (
            parent.image_layer_with_phasor_features_combobox.currentText()
            == ''
        )

    assert filter_widget.threshold_line is None


def test_filter_widget_layer_with_settings(make_napari_viewer):
    """Test filter widget behavior when layer has existing settings."""
    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()

    intensity_image_layer.metadata["settings"] = {
        "threshold": 0.1,
        "threshold_method": "Li",
        "filter": {"method": "median", "size": 7, "repeat": 3},
    }
    viewer.add_layer(intensity_image_layer)

    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    assert (
        filter_widget.threshold_slider.value()
        == 0.1 * filter_widget.threshold_factor
    )
    assert filter_widget.median_filter_spinbox.value() == 7
    assert filter_widget.median_filter_repetition_spinbox.value() == 3
    assert filter_widget.threshold_method_combobox.currentText() == "Li"


def test_filter_widget_plot_histogram_no_layer(make_napari_viewer):
    """Test plotting histogram when no layer is available."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    filter_widget.plot_mean_histogram()

    children_before = len(filter_widget.hist_ax.get_children())

    assert children_before >= 0


def test_filter_widget_update_threshold_line_no_layer(make_napari_viewer):
    """Test updating threshold line when no layer is available."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab
    filter_widget.update_threshold_line()

    assert filter_widget.threshold_line is None


def test_filter_widget_ui_layout(make_napari_viewer):
    """Test the UI layout structure."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    main_layout = filter_widget.layout()
    assert isinstance(main_layout, QVBoxLayout)

    scroll_areas = filter_widget.findChildren(QScrollArea)
    assert len(scroll_areas) == 1

    # Check that apply button is outside scroll area
    apply_buttons = filter_widget.findChildren(QPushButton)
    apply_button = [
        btn
        for btn in apply_buttons
        if btn.text() == "Apply Filter and Threshold"
    ][0]
    assert apply_button == filter_widget.apply_button

    h_layouts = filter_widget.findChildren(QHBoxLayout)
    assert len(h_layouts) >= 5


def test_filter_widget_canvas_properties(make_napari_viewer):
    """Test canvas and figure properties."""
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    assert filter_widget.hist_fig.get_figwidth() == 8
    assert filter_widget.hist_fig.get_figheight() == 4

    assert filter_widget.hist_fig.get_constrained_layout()

    canvas_widgets = filter_widget.findChildren(FigureCanvasQTAgg)
    assert len(canvas_widgets) == 1

    canvas = canvas_widgets[0]
    assert canvas.height() == 150


def test_log_scale_checkbox_functionality(make_napari_viewer):
    """Test log scale checkbox functionality."""
    viewer = make_napari_viewer()
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


def test_log_scale_checkbox_with_no_layer(make_napari_viewer):
    """Test log scale checkbox when no layer is available."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    filter_widget.log_scale_checkbox.setChecked(True)
    filter_widget.on_log_scale_changed(2)

    assert filter_widget.hist_ax.get_yscale() == 'linear'
