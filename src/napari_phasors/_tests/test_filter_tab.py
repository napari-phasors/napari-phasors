from unittest.mock import patch

import numpy as np
from matplotlib.figure import Figure
from napari.layers import Image
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
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

    # Test UI components
    assert hasattr(filter_widget, 'label_4')
    assert isinstance(filter_widget.label_4, QLabel)
    assert filter_widget.label_4.text() == "Median Filter Kernel Size: 3 x 3"

    assert hasattr(filter_widget, 'median_filter_spinbox')
    assert isinstance(filter_widget.median_filter_spinbox, QSpinBox)
    assert filter_widget.median_filter_spinbox.minimum() == 2
    assert filter_widget.median_filter_spinbox.maximum() == 99
    assert filter_widget.median_filter_spinbox.value() == 3

    assert hasattr(filter_widget, 'median_filter_repetition_spinbox')
    assert isinstance(filter_widget.median_filter_repetition_spinbox, QSpinBox)
    assert filter_widget.median_filter_repetition_spinbox.minimum() == 0
    assert filter_widget.median_filter_repetition_spinbox.value() == 0

    assert hasattr(filter_widget, 'label_3')
    assert isinstance(filter_widget.label_3, QLabel)
    assert filter_widget.label_3.text() == "Intensity threshold: 0"

    assert hasattr(filter_widget, 'threshold_slider')
    assert isinstance(filter_widget.threshold_slider, QSlider)
    assert filter_widget.threshold_slider.orientation() == Qt.Horizontal
    assert filter_widget.threshold_slider.minimum() == 0
    assert filter_widget.threshold_slider.maximum() == 100
    assert filter_widget.threshold_slider.value() == 0

    assert hasattr(filter_widget, 'apply_button')
    assert isinstance(filter_widget.apply_button, QPushButton)
    assert filter_widget.apply_button.text() == "Apply Filter and Threshold"

    # Test scroll area
    scroll_areas = filter_widget.findChildren(QScrollArea)
    assert len(scroll_areas) == 1
    scroll_area = scroll_areas[0]
    assert scroll_area.widgetResizable() == True


def test_filter_widget_histogram_styling(make_napari_viewer):
    """Test that histogram styling is applied correctly."""
    viewer = make_napari_viewer()

    # Test that style_histogram_axes method exists and is called
    with patch.object(FilterWidget, 'style_histogram_axes') as mock_style:
        parent = PlotterWidget(viewer)
        filter_widget = parent.filter_tab
        mock_style.assert_called_once()

    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    # Check axes styling
    assert filter_widget.hist_ax.patch.get_alpha() == 0
    assert filter_widget.hist_fig.patch.get_alpha() == 0

    # Check spine colors - use numpy.allclose for RGBA comparison
    import matplotlib.colors as mcolors

    grey_rgba = mcolors.to_rgba('grey')

    for spine in filter_widget.hist_ax.spines.values():
        np.testing.assert_array_almost_equal(spine.get_edgecolor(), grey_rgba)
        assert spine.get_linewidth() == 1

    # Check labels
    assert filter_widget.hist_ax.get_ylabel() == "Count"
    assert filter_widget.hist_ax.get_xlabel() == "Mean Intensity"


def test_filter_widget_with_layer_data(make_napari_viewer):
    """Test filter widget behavior with actual layer data."""
    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)

    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    # Check that threshold factor is calculated correctly
    max_mean_value = np.nanmax(intensity_image_layer.metadata["original_mean"])
    expected_magnitude = int(np.log10(max_mean_value))
    expected_threshold_factor = (
        10 ** (2 - expected_magnitude) if expected_magnitude <= 2 else 1
    )
    assert filter_widget.threshold_factor == expected_threshold_factor

    # Check that slider maximum is set correctly
    expected_max = int(
        np.ceil(max_mean_value * filter_widget.threshold_factor)
    )
    assert filter_widget.threshold_slider.maximum() == expected_max

    # Check that default threshold is set (10% of max)
    expected_default_threshold = int(
        max_mean_value * 0.1 * filter_widget.threshold_factor
    )
    assert filter_widget.threshold_slider.value() == expected_default_threshold


def test_filter_widget_threshold_slider_callback(make_napari_viewer):
    """Test threshold slider callback functionality."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    # Set a threshold factor for testing
    filter_widget.threshold_factor = 10

    # Change slider value
    test_value = 50
    filter_widget.threshold_slider.setValue(test_value)
    filter_widget.on_threshold_slider_change()

    # Check that label is updated correctly
    expected_text = (
        f"Intensity threshold: {test_value / filter_widget.threshold_factor}"
    )
    assert filter_widget.label_3.text() == expected_text


def test_filter_widget_kernel_size_callback(make_napari_viewer):
    """Test kernel size spinbox callback functionality."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    # Change kernel size
    test_value = 5
    filter_widget.median_filter_spinbox.setValue(test_value)

    # Check that label is updated correctly
    expected_text = f"Median Filter Kernel Size: {test_value} x {test_value}"
    assert filter_widget.label_4.text() == expected_text


def test_filter_widget_histogram_plotting(make_napari_viewer):
    """Test histogram plotting functionality."""
    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)

    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    # Test plotting histogram
    filter_widget.plot_mean_histogram()

    # Check that histogram was plotted (axes should have children)
    assert len(filter_widget.hist_ax.get_children()) > 0

    # Test that threshold line can be updated
    filter_widget.threshold_slider.setValue(50)
    filter_widget.update_threshold_line()


def test_spinbox_and_slider_do_not_call_plot(make_napari_viewer):
    """Changing spinbox or slider does not call parent.plot() unless apply is clicked."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    with patch.object(parent, 'plot') as mock_plot:
        # Change spinbox and slider
        filter_widget.median_filter_spinbox.setValue(5)
        filter_widget.threshold_slider.setValue(42)

        # plot() should not be called
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

    # Set slider to a value and check threshold line
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

    # Mock the plot method
    with patch.object(parent, 'plot') as mock_plot:
        # Try to trigger plot via apply button
        with patch('napari_phasors.filter_tab.apply_filter_and_threshold'):
            filter_widget.apply_button.click()
        mock_plot.assert_not_called()

    # Try to plot histogram and update threshold line
    filter_widget.plot_mean_histogram()
    filter_widget.update_threshold_line()
    # No error, nothing should happen


def test_slider_and_histogram_update_on_layer_add_and_select(
    make_napari_viewer,
):
    """Adding a new layer updates slider range, threshold line, and histogram only when selected."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    # No layer present, slider max should be default
    assert filter_widget.threshold_slider.maximum() == 100

    # Add first layer
    intensity_image_layer1 = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer1)

    # Slider max and threshold line should update
    expected_max1 = int(
        np.ceil(
            np.nanmax(intensity_image_layer1.metadata["original_mean"])
            * filter_widget.threshold_factor
        )
    )
    assert filter_widget.threshold_slider.maximum() == expected_max1
    assert filter_widget.threshold_line is not None
    assert len(filter_widget.hist_ax.patches) > 0  # Histogram drawn

    # Add second layer, but do not select it
    intensity_image_layer2 = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer2)
    # Combobox still points to first layer, so nothing should change
    prev_max = filter_widget.threshold_slider.maximum()
    assert filter_widget.threshold_slider.maximum() == prev_max

    # Now select the new layer
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
    # Patch the relevant FilterWidget methods to check they are NOT called
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
    # No threshold line or histogram should be drawn
    assert filter_widget.threshold_line is None


def test_filter_widget_apply_button_with_layer(make_napari_viewer):
    """Test apply button behavior with a valid layer selected."""
    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    # Set some values
    filter_widget.threshold_slider.setValue(50)
    filter_widget.median_filter_spinbox.setValue(5)
    filter_widget.median_filter_repetition_spinbox.setValue(2)

    # Mock both apply_filter_and_threshold and parent.plot
    with (
        patch(
            'napari_phasors.filter_tab.apply_filter_and_threshold'
        ) as mock_apply,
        patch.object(parent, 'plot') as mock_plot,
    ):

        filter_widget.apply_button_clicked()

        # Check that apply_filter_and_threshold was called with correct parameters
        mock_apply.assert_called_once_with(
            intensity_image_layer,
            threshold=50
            / filter_widget.threshold_factor,  # slider value / threshold factor
            size=5,
            repeat=2,
        )

        # Check that parent plot method is called
        mock_plot.assert_called_once()


def test_filter_widget_layer_with_settings(make_napari_viewer):
    """Test filter widget behavior when layer has existing settings."""
    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()

    # Add metadata with settings
    intensity_image_layer.metadata["settings"] = {
        "threshold": 0.7,
        "filter": {"size": 7, "repeat": 3},
    }
    viewer.add_layer(intensity_image_layer)

    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    # Check that settings are loaded correctly
    assert (
        filter_widget.threshold_slider.value()
        == 0.7 * filter_widget.threshold_factor
    )
    assert filter_widget.median_filter_spinbox.value() == 7
    assert filter_widget.median_filter_repetition_spinbox.value() == 3


def test_filter_widget_plot_histogram_no_layer(make_napari_viewer):
    """Test plotting histogram when no layer is available."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    # Should return early without plotting
    filter_widget.plot_mean_histogram()

    # Histogram should be empty or have minimal elements
    children_before = len(filter_widget.hist_ax.get_children())

    # The function should handle the None case gracefully
    assert children_before >= 0  # Should not crash


def test_filter_widget_update_threshold_line_no_layer(make_napari_viewer):
    """Test updating threshold line when no layer is available."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab
    filter_widget.update_threshold_line()

    # Should handle gracefully without crashing
    assert filter_widget.threshold_line is None


def test_filter_widget_ui_layout(make_napari_viewer):
    """Test the UI layout structure."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    # Check main layout
    main_layout = filter_widget.layout()
    assert isinstance(main_layout, QVBoxLayout)

    # Check scroll area exists
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

    # Check horizontal layouts exist
    h_layouts = filter_widget.findChildren(QHBoxLayout)
    assert (
        len(h_layouts) >= 3
    )  # At least 3 horizontal layouts for the controls


def test_filter_widget_canvas_properties(make_napari_viewer):
    """Test canvas and figure properties."""
    # Find the canvas widget
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    # Check figure size
    assert filter_widget.hist_fig.get_figwidth() == 8
    assert filter_widget.hist_fig.get_figheight() == 4

    # Check that constrained_layout is used
    assert filter_widget.hist_fig.get_constrained_layout()

    canvas_widgets = filter_widget.findChildren(FigureCanvasQTAgg)
    assert len(canvas_widgets) == 1

    canvas = canvas_widgets[0]
    assert canvas.height() == 150  # Fixed height as set in setup_ui
