from unittest.mock import MagicMock, patch

import matplotlib.colors as mcolors
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from napari.layers import Image
from phasorpy.lifetime import (
    phasor_to_apparent_lifetime,
    phasor_to_normal_lifetime,
)
from phasorpy.phasor import phasor_to_polar
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QVBoxLayout,
)
from superqt import QRangeSlider

from napari_phasors._tests.test_plotter import create_image_layer_with_phasors
from napari_phasors._utils import HistogramWidget
from napari_phasors.plotter import PlotterWidget


def test_phasor_mapping_widget_initialization_values(make_napari_viewer):
    """Test the initialization of the Lifetime Widget."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    lifetime_widget = parent.phasor_mapping_tab

    # Basic widget structure tests
    assert lifetime_widget.viewer == viewer
    assert lifetime_widget.parent_widget == parent
    assert isinstance(lifetime_widget.layout(), QVBoxLayout)

    # Test initial attribute values
    assert lifetime_widget.frequency is None
    assert lifetime_widget.lifetime_data is None
    assert lifetime_widget.lifetime_data_original is None
    assert lifetime_widget.lifetime_layer is None
    assert lifetime_widget.min_lifetime is None
    assert lifetime_widget.max_lifetime is None
    assert lifetime_widget.lifetime_colormap is None
    assert lifetime_widget.colormap_contrast_limits is None
    assert lifetime_widget.lifetime_type is None
    assert lifetime_widget.lifetime_range_factor == 1000
    assert lifetime_widget.histogram_widget._slider_being_dragged is False

    # Test histogram widget initialization
    assert isinstance(lifetime_widget.histogram_widget, HistogramWidget)
    assert isinstance(lifetime_widget.histogram_widget.fig, Figure)
    assert lifetime_widget.histogram_widget.ax is not None

    # Test UI components
    assert hasattr(lifetime_widget, 'frequency_input')
    assert isinstance(lifetime_widget.frequency_input, QLineEdit)

    assert hasattr(lifetime_widget, 'lifetime_type_combobox')
    assert isinstance(lifetime_widget.lifetime_type_combobox, QComboBox)
    expected_items = [
        "Apparent Phase Lifetime",
        "Apparent Modulation Lifetime",
        "Normal Lifetime",
    ]
    actual_items = [
        lifetime_widget.lifetime_type_combobox.itemText(i)
        for i in range(lifetime_widget.lifetime_type_combobox.count())
    ]
    assert actual_items == expected_items
    assert (
        lifetime_widget.lifetime_type_combobox.currentText()
        == "Apparent Phase Lifetime"
    )

    assert hasattr(lifetime_widget, 'lifetime_range_label')
    assert isinstance(lifetime_widget.lifetime_range_label, QLabel)
    assert (
        lifetime_widget.lifetime_range_label.text()
        == "Lifetime range (ns): 0.0 - 100.0"
    )

    assert hasattr(lifetime_widget, 'lifetime_min_edit')
    assert isinstance(lifetime_widget.lifetime_min_edit, QLineEdit)
    assert lifetime_widget.lifetime_min_edit.text() == "0.0"

    assert hasattr(lifetime_widget, 'lifetime_max_edit')
    assert isinstance(lifetime_widget.lifetime_max_edit, QLineEdit)
    assert lifetime_widget.lifetime_max_edit.text() == "100.0"

    assert hasattr(lifetime_widget, 'lifetime_range_slider')
    assert isinstance(lifetime_widget.lifetime_range_slider, QRangeSlider)
    assert (
        lifetime_widget.lifetime_range_slider.orientation()
        == Qt.Orientation.Horizontal
    )
    assert lifetime_widget.lifetime_range_slider.minimum() == 0
    assert lifetime_widget.lifetime_range_slider.maximum() == 100
    assert lifetime_widget.lifetime_range_slider.value() == (0, 100)

    # Test scroll area
    scroll_areas = lifetime_widget.findChildren(QScrollArea)
    assert len(scroll_areas) == 1
    scroll_area = scroll_areas[0]
    assert scroll_area.widgetResizable()

    # Histogram widget is now hosted in the shared dock stack.
    assert (
        parent.phasor_map_histogram_dock_widget.histogram_widget
        is lifetime_widget.histogram_widget
    )
    assert (
        parent._histogram_stack.indexOf(
            parent.phasor_map_histogram_dock_widget
        )
        >= 0
    )


def test_phasor_mapping_widget_histogram_styling(make_napari_viewer):
    """Test that histogram styling is applied correctly."""
    viewer = make_napari_viewer()

    parent = PlotterWidget(viewer)
    lifetime_widget = parent.phasor_mapping_tab

    # Check axes styling via the HistogramWidget
    assert lifetime_widget.histogram_widget.ax.patch.get_alpha() == 0
    assert lifetime_widget.histogram_widget.fig.patch.get_alpha() == 0

    # Check spine colors - use numpy.allclose for RGBA comparison
    grey_rgba = mcolors.to_rgba('grey')

    for spine in lifetime_widget.histogram_widget.ax.spines.values():
        np.testing.assert_array_almost_equal(spine.get_edgecolor(), grey_rgba)
        assert spine.get_linewidth() == 1

    # Check labels
    assert lifetime_widget.histogram_widget.ax.get_ylabel() == "Pixel count"
    assert lifetime_widget.histogram_widget.ax.get_xlabel() == "Lifetime (ns)"


def test_phasor_mapping_widget_frequency_input_validation(make_napari_viewer):
    """Test frequency input validation."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    lifetime_widget = parent.phasor_mapping_tab

    # Test that frequency input has double validator
    validator = lifetime_widget.frequency_input.validator()
    assert validator is not None

    # Test valid input
    lifetime_widget.frequency_input.setText("80.0")
    assert lifetime_widget.frequency_input.text() == "80.0"


def test_phasor_mapping_widget_slider_drag_state(make_napari_viewer):
    """Test slider drag state tracking."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    lifetime_widget = parent.phasor_mapping_tab

    # Initially not being dragged
    assert lifetime_widget.histogram_widget._slider_being_dragged is False

    # Simulate slider press
    lifetime_widget.histogram_widget._on_slider_pressed()
    assert lifetime_widget.histogram_widget._slider_being_dragged is True

    # Simulate slider release
    lifetime_widget.histogram_widget._on_slider_released()
    assert lifetime_widget.histogram_widget._slider_being_dragged is False


def test_phasor_mapping_widget_range_label_update(make_napari_viewer):
    """Test lifetime range label update."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    lifetime_widget = parent.phasor_mapping_tab

    # Test label update
    test_value = (25000, 75000)  # Represents 25.0 - 75.0 ns with factor 1000
    lifetime_widget.histogram_widget._on_range_label_update(test_value)

    assert (
        lifetime_widget.lifetime_range_label.text()
        == "Lifetime range (ns): 25.00 - 75.00"
    )
    assert lifetime_widget.lifetime_min_edit.text() == "25.00"
    assert lifetime_widget.lifetime_max_edit.text() == "75.00"


def test_phasor_mapping_widget_calculate_lifetimes_no_layer(
    make_napari_viewer,
):
    """Test calculate_lifetimes when no layer is available."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    lifetime_widget = parent.phasor_mapping_tab

    # Should return early without error
    lifetime_widget.calculate_lifetimes()
    assert lifetime_widget.lifetime_data_original is None


def test_phasor_mapping_widget_plot_histogram_no_data(make_napari_viewer):
    """Test plotting histogram when no data is available."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    lifetime_widget = parent.phasor_mapping_tab

    # No data should leave the histogram empty with controls disabled.
    lifetime_widget.plot_lifetime_histogram()
    assert lifetime_widget.histogram_widget.counts is None
    assert not lifetime_widget.histogram_widget._settings_button.isEnabled()
    assert not lifetime_widget.histogram_widget.save_png_button.isEnabled()


def test_phasor_mapping_widget_ui_layout(make_napari_viewer):
    """Test the UI layout structure."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    lifetime_widget = parent.phasor_mapping_tab

    # Check main layout
    main_layout = lifetime_widget.layout()
    assert isinstance(main_layout, QVBoxLayout)

    # Check scroll area exists
    scroll_areas = lifetime_widget.findChildren(QScrollArea)
    assert len(scroll_areas) == 1

    # Check horizontal layouts exist for range controls
    h_layouts = lifetime_widget.findChildren(QHBoxLayout)
    assert len(h_layouts) >= 1  # At least one for the min/max edit controls


def test_phasor_mapping_widget_canvas_properties(make_napari_viewer):
    """Test canvas and figure properties."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    lifetime_widget = parent.phasor_mapping_tab

    # Check figure size
    assert lifetime_widget.histogram_widget.fig.get_figwidth() == 8
    assert lifetime_widget.histogram_widget.fig.get_figheight() == 4

    # Check that constrained_layout is used
    assert lifetime_widget.histogram_widget.fig.get_constrained_layout()

    canvas_widgets = lifetime_widget.findChildren(FigureCanvasQTAgg)
    # The histogram canvas now lives in the detachable dock widget,
    # not inside the lifetime tab itself.
    assert len(canvas_widgets) == 0

    # Access the canvas through the histogram widget directly
    canvas = lifetime_widget.histogram_widget.fig.canvas
    assert isinstance(canvas, FigureCanvasQTAgg)
    assert canvas.height() == 150  # Fixed height as set in setup_ui


def test_phasor_mapping_widget_type_changed_no_frequency(make_napari_viewer):
    """Test behavior when Calculate is clicked but no frequency is set."""
    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)

    parent = PlotterWidget(viewer)
    lifetime_widget = parent.phasor_mapping_tab

    # Set a lifetime type but don't set frequency
    lifetime_widget.lifetime_type_combobox.setCurrentText(
        "Apparent Phase Lifetime"
    )

    with patch(
        'napari_phasors.phasor_mapping_tab.show_warning'
    ) as mock_warning:
        lifetime_widget._on_calculate_lifetime_clicked()
        mock_warning.assert_called_once_with("Enter frequency")


def test_phasor_mapping_widget_settings_initialization_in_metadata(
    make_napari_viewer,
):
    """Test that lifetime settings are only initialized when analysis is performed."""
    viewer = make_napari_viewer()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)
    parent = PlotterWidget(viewer)
    lifetime_widget = parent.phasor_mapping_tab

    # Select the layer
    parent.image_layer_with_phasor_features_combobox.setCurrentText(layer.name)
    lifetime_widget._on_image_layer_changed()

    # Check that lifetime settings were NOT initialized
    if 'settings' in layer.metadata:
        assert 'lifetime' not in layer.metadata['settings']

    # Now perform lifetime analysis
    lifetime_widget.frequency_input.setText("80.0")
    lifetime_widget._on_frequency_changed()

    # Select a lifetime type
    lifetime_widget.lifetime_type_combobox.setCurrentText(
        "Apparent Phase Lifetime"
    )

    # Click Calculate to trigger analysis and initialize metadata
    lifetime_widget._on_calculate_lifetime_clicked()

    # Now check that settings were initialized
    assert 'settings' in layer.metadata
    assert 'lifetime' in layer.metadata['settings']
    assert 'frequency' in layer.metadata['settings']

    # Check values
    assert layer.metadata['settings']['frequency'] == 80.0
    assert (
        layer.metadata['settings']['lifetime']['lifetime_type']
        == 'Apparent Phase Lifetime'
    )
    # Range values should be set after calculation
    assert 'lifetime_range_min' in layer.metadata['settings']['lifetime']
    assert 'lifetime_range_max' in layer.metadata['settings']['lifetime']

    parent.deleteLater()


def test_phasor_mapping_widget_settings_update_in_metadata(make_napari_viewer):
    """Test that changing settings updates layer metadata."""
    viewer = make_napari_viewer()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)
    parent = PlotterWidget(viewer)
    lifetime_widget = parent.phasor_mapping_tab

    # Set frequency
    lifetime_widget.frequency_input.setText("80.0")
    lifetime_widget._on_frequency_changed()
    parent._broadcast_frequency_value_across_tabs("80.0")
    assert layer.metadata['settings']['frequency'] == 80.0

    # Set lifetime type
    lifetime_widget.lifetime_type_combobox.setCurrentText(
        "Apparent Phase Lifetime"
    )
    assert (
        layer.metadata['settings']['lifetime']['lifetime_type']
        == 'Apparent Phase Lifetime'
    )

    # Set lifetime range
    lifetime_widget.lifetime_range_slider.setValue((1000, 5000))
    lifetime_widget._on_lifetime_range_changed((1000, 5000))
    assert layer.metadata['settings']['lifetime']['lifetime_range_min'] == 1.0
    assert layer.metadata['settings']['lifetime']['lifetime_range_max'] == 5.0

    parent.deleteLater()


def test_phasor_mapping_widget_settings_persistence_across_layer_switches(
    make_napari_viewer,
):
    """Test that settings persist when switching between layers."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    lifetime_widget = parent.phasor_mapping_tab

    # Create two layers
    layer_1 = create_image_layer_with_phasors()
    layer_2 = create_image_layer_with_phasors()
    viewer.add_layer(layer_1)
    viewer.add_layer(layer_2)

    # Ensure layer_2 is currently active (it's the last one added)
    parent.image_layer_with_phasor_features_combobox.setCurrentText(
        layer_2.name
    )
    lifetime_widget._on_image_layer_changed()

    # Layer 2 should start with defaults
    assert lifetime_widget.frequency_input.text() == ""
    assert (
        lifetime_widget.lifetime_type_combobox.currentText()
        == 'Apparent Phase Lifetime'
    )

    # Now switch to layer_1
    parent.image_layer_with_phasor_features_combobox.setCurrentText(
        layer_1.name
    )
    lifetime_widget._on_image_layer_changed()

    # Modify settings for layer_1
    lifetime_widget.frequency_input.setText("80.0")
    # Manually trigger the broadcast since we're setting it programmatically
    parent._broadcast_frequency_value_across_tabs("80.0")

    lifetime_widget.lifetime_type_combobox.setCurrentText("Normal Lifetime")

    # Click Calculate to trigger the analysis and save to metadata
    lifetime_widget._on_calculate_lifetime_clicked()

    # Verify settings are saved in layer_1 metadata
    assert layer_1.metadata['settings']['frequency'] == 80.0
    assert (
        layer_1.metadata['settings']['lifetime']['lifetime_type']
        == 'Normal Lifetime'
    )

    # Switch to layer_2 (should have defaults)
    parent.image_layer_with_phasor_features_combobox.setCurrentText(
        layer_2.name
    )
    lifetime_widget._on_image_layer_changed()

    assert lifetime_widget.frequency_input.text() == ""
    assert (
        lifetime_widget.lifetime_type_combobox.currentText()
        == 'Apparent Phase Lifetime'
    )

    # Switch back to layer_1 (should restore settings)
    parent.image_layer_with_phasor_features_combobox.setCurrentText(
        layer_1.name
    )
    lifetime_widget._on_image_layer_changed()

    assert lifetime_widget.frequency_input.text() == "80.0"
    assert (
        lifetime_widget.lifetime_type_combobox.currentText()
        == 'Normal Lifetime'
    )

    parent.deleteLater()


def test_phasor_mapping_widget_adding_layer_without_settings_initializes_defaults(
    make_napari_viewer,
):
    """Test that adding a layer without lifetime settings doesn't auto-initialize metadata."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    lifetime_widget = parent.phasor_mapping_tab

    # Create a layer and remove lifetime settings if they exist
    layer = create_image_layer_with_phasors()
    if (
        'settings' in layer.metadata
        and 'lifetime' in layer.metadata['settings']
    ):
        del layer.metadata['settings']['lifetime']

    viewer.add_layer(layer)

    # Trigger layer change - should NOT initialize lifetime metadata
    parent.image_layer_with_phasor_features_combobox.setCurrentText(layer.name)
    lifetime_widget._on_image_layer_changed()

    # Verify settings were NOT auto-initialized
    if 'settings' in layer.metadata:
        assert 'lifetime' not in layer.metadata['settings']

    # Now perform actual lifetime analysis
    lifetime_widget.frequency_input.setText("80.0")
    lifetime_widget._on_frequency_changed()

    # Select a lifetime type
    lifetime_widget.lifetime_type_combobox.setCurrentText(
        "Apparent Phase Lifetime"
    )

    # Click Calculate to trigger analysis and initialize metadata
    lifetime_widget._on_calculate_lifetime_clicked()

    # Now verify settings were initialized with actual values
    assert 'settings' in layer.metadata
    assert 'lifetime' in layer.metadata['settings']
    assert layer.metadata['settings']['frequency'] == 80.0
    assert (
        layer.metadata['settings']['lifetime']['lifetime_type']
        == 'Apparent Phase Lifetime'
    )
    # Range values should be set after calculation
    assert 'lifetime_range_min' in layer.metadata['settings']['lifetime']
    assert 'lifetime_range_max' in layer.metadata['settings']['lifetime']

    parent.deleteLater()


def test_phasor_mapping_widget_settings_restored_after_recalculation(
    make_napari_viewer,
):
    """Test that lifetime range settings are restored after recalculating lifetimes."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    lifetime_widget = parent.phasor_mapping_tab

    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)

    # Set frequency and lifetime type
    lifetime_widget.frequency_input.setText("80.0")
    lifetime_widget._on_frequency_changed()
    lifetime_widget.lifetime_type_combobox.setCurrentText(
        "Apparent Phase Lifetime"
    )

    # Click Calculate to trigger the calculation
    lifetime_widget._on_calculate_lifetime_clicked()

    # Wait for calculation to complete
    assert lifetime_widget.lifetime_data is not None

    # Set custom range
    custom_min = 2.0
    custom_max = 4.0
    min_slider = int(custom_min * lifetime_widget.lifetime_range_factor)
    max_slider = int(custom_max * lifetime_widget.lifetime_range_factor)

    lifetime_widget.lifetime_range_slider.setValue((min_slider, max_slider))
    lifetime_widget._on_lifetime_range_changed((min_slider, max_slider))

    # Verify range is saved in metadata
    assert (
        abs(
            layer.metadata['settings']['lifetime']['lifetime_range_min']
            - custom_min
        )
        < 0.01
    )
    assert (
        abs(
            layer.metadata['settings']['lifetime']['lifetime_range_max']
            - custom_max
        )
        < 0.01
    )

    # Change to different lifetime type and recalculate
    lifetime_widget.lifetime_type_combobox.setCurrentText(
        "Apparent Modulation Lifetime"
    )
    lifetime_widget._on_calculate_lifetime_clicked()

    # Change back and recalculate
    lifetime_widget.lifetime_type_combobox.setCurrentText(
        "Apparent Phase Lifetime"
    )
    lifetime_widget._on_calculate_lifetime_clicked()

    # Range should be restored from metadata
    assert (
        abs(float(lifetime_widget.lifetime_min_edit.text()) - custom_min)
        < 0.01
    )
    assert (
        abs(float(lifetime_widget.lifetime_max_edit.text()) - custom_max)
        < 0.01
    )

    parent.deleteLater()


def test_phasor_mapping_widget_adding_removing_layers_updates_settings(
    make_napari_viewer,
):
    """Test that adding/removing layers properly manages settings."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    lifetime_widget = parent.phasor_mapping_tab

    # Add first layer with settings
    layer_1 = create_image_layer_with_phasors()
    viewer.add_layer(layer_1)
    parent.image_layer_with_phasor_features_combobox.setCurrentText(
        layer_1.name
    )

    lifetime_widget.frequency_input.setText("80.0")
    parent._broadcast_frequency_value_across_tabs("80.0")
    lifetime_widget._on_frequency_changed()
    lifetime_widget.lifetime_type_combobox.setCurrentText("Normal Lifetime")

    # Click Calculate to trigger analysis
    lifetime_widget._on_calculate_lifetime_clicked()

    # Check settings were saved
    assert layer_1.metadata['settings']['frequency'] == 80.0
    assert (
        layer_1.metadata['settings']['lifetime']['lifetime_type']
        == 'Normal Lifetime'
    )

    # Add second layer
    layer_2 = create_image_layer_with_phasors()
    viewer.add_layer(layer_2)
    parent.image_layer_with_phasor_features_combobox.setCurrentText(
        layer_2.name
    )
    lifetime_widget._on_image_layer_changed()

    # Layer 2 should have defaults
    assert (
        lifetime_widget.lifetime_type_combobox.currentText()
        == 'Apparent Phase Lifetime'
    )

    # Remove layer 1
    viewer.layers.remove(layer_1)

    # Layer 2 should still be selectable and have defaults
    parent.image_layer_with_phasor_features_combobox.setCurrentText(
        layer_2.name
    )
    lifetime_widget._on_image_layer_changed()
    assert (
        lifetime_widget.lifetime_type_combobox.currentText()
        == 'Apparent Phase Lifetime'
    )

    parent.deleteLater()


def test_phasor_mapping_widget_frequency_saved_on_lifetime_type_change(
    make_napari_viewer,
):
    """Test that frequency is saved to metadata when Calculate is clicked."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    lifetime_widget = parent.phasor_mapping_tab

    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)

    # Set frequency
    lifetime_widget.frequency_input.setText("80.0")
    lifetime_widget._on_frequency_changed()

    # Change lifetime type
    lifetime_widget.lifetime_type_combobox.setCurrentText(
        "Apparent Phase Lifetime"
    )

    # Click Calculate to trigger analysis and save frequency to metadata
    lifetime_widget._on_calculate_lifetime_clicked()

    # Check frequency is in general settings
    assert layer.metadata['settings']['frequency'] == 80.0

    parent.deleteLater()


def test_phasor_mapping_widget_no_recursive_updates_when_restoring_settings(
    make_napari_viewer,
):
    """Test that restoring settings doesn't trigger recursive updates."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    lifetime_widget = parent.phasor_mapping_tab

    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)

    # Set up initial state
    lifetime_widget.frequency_input.setText("80.0")
    lifetime_widget._on_frequency_changed()
    lifetime_widget.lifetime_type_combobox.setCurrentText(
        "Apparent Phase Lifetime"
    )

    # Mock the update method to check it's not called during restoration
    with patch.object(lifetime_widget, '_update_lifetime_setting_in_metadata'):
        # Switch to another layer and back (triggers restoration)
        layer_2 = create_image_layer_with_phasors()
        viewer.add_layer(layer_2)
        parent.image_layer_with_phasor_features_combobox.setCurrentText(
            layer_2.name
        )
        lifetime_widget._on_image_layer_changed()

        parent.image_layer_with_phasor_features_combobox.setCurrentText(
            layer.name
        )
        lifetime_widget._on_image_layer_changed()

        # _update_lifetime_setting_in_metadata should not be called during restoration
        # because _updating_settings flag should prevent it
        # We can't easily test this without checking the flag behavior, but we can verify
        # that the settings were restored correctly
        assert (
            lifetime_widget.lifetime_type_combobox.currentText()
            == 'Apparent Phase Lifetime'
        )

    parent.deleteLater()


def test_phasor_mapping_widget_slider_range_update(make_napari_viewer):
    """Test updating slider range based on lifetime data."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    lifetime_widget = parent.phasor_mapping_tab

    # Create mock lifetime data
    lifetime_widget.lifetime_data_original = np.array(
        [1.0, 2.0, 3.0, 4.0, 5.0]
    )
    lifetime_widget.frequency = 80.0  # MHz

    lifetime_widget._update_lifetime_range_slider()

    # Check that min/max are set correctly
    assert lifetime_widget.min_lifetime == 1.0
    assert lifetime_widget.max_lifetime == 5.0

    # Check slider range
    assert lifetime_widget.lifetime_range_slider.minimum() == 0
    assert (
        lifetime_widget.lifetime_range_slider.maximum() == 5000
    )  # 5.0 * 1000
    assert lifetime_widget.lifetime_range_slider.value() == (
        1000,
        5000,
    )  # (1.0 * 1000, 5.0 * 1000)


def test_phasor_mapping_widget_slider_range_update_no_valid_data(
    make_napari_viewer,
):
    """Test updating slider range when no valid data exists."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    lifetime_widget = parent.phasor_mapping_tab

    # Create data with only invalid values
    lifetime_widget.lifetime_data_original = np.array([np.nan, 0, np.inf, -1])
    lifetime_widget.frequency = 80.0  # MHz

    lifetime_widget._update_lifetime_range_slider()

    # Check that defaults are used
    assert lifetime_widget.min_lifetime == 0.0
    assert lifetime_widget.max_lifetime == 10.0
    assert lifetime_widget.lifetime_range_slider.maximum() == 10000


def test_phasor_mapping_widget_min_max_edit_callbacks(make_napari_viewer):
    """Test manual entry of min/max values."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    lifetime_widget = parent.phasor_mapping_tab

    # Set some initial max lifetime for validation
    lifetime_widget.max_lifetime = 10.0

    # Test min edit
    lifetime_widget.lifetime_min_edit.setText("2.5")
    lifetime_widget.lifetime_max_edit.setText("7.5")

    with patch.object(
        lifetime_widget, '_on_lifetime_range_changed'
    ) as mock_range_changed:
        lifetime_widget.histogram_widget._on_range_min_edit()
        mock_range_changed.assert_called_once()

    with patch.object(
        lifetime_widget, '_on_lifetime_range_changed'
    ) as mock_range_changed:
        lifetime_widget.histogram_widget._on_range_max_edit()
        mock_range_changed.assert_called_once()


def test_phasor_mapping_widget_image_layer_changed_with_settings(
    make_napari_viewer,
):
    """Test behavior when image layer changes and has frequency settings."""
    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()
    intensity_image_layer.metadata["settings"] = {"frequency": 80.0}
    viewer.add_layer(intensity_image_layer)

    parent = PlotterWidget(viewer)
    lifetime_widget = parent.phasor_mapping_tab

    # Mock the harmonic value
    parent.harmonic = 1

    lifetime_widget._on_image_layer_changed()

    # Check that frequency is loaded from settings
    assert lifetime_widget.frequency_input.text() == "80.0"


def test_phasor_mapping_widget_image_layer_changed_no_layer(
    make_napari_viewer,
):
    """Test behavior when no layer is selected."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    lifetime_widget = parent.phasor_mapping_tab

    # Mock empty layer name
    parent.image_layer_with_phasor_features_combobox.setCurrentText("")

    lifetime_widget._on_image_layer_changed()

    # Histogram should be reset when no layer is selected.
    assert lifetime_widget.lifetime_data is None
    assert lifetime_widget.lifetime_data_original is None
    assert lifetime_widget.histogram_widget.counts is None
    assert not lifetime_widget.histogram_widget._settings_button.isEnabled()
    assert not lifetime_widget.histogram_widget.save_png_button.isEnabled()


def test_phasor_mapping_widget_colormap_changed_callback(make_napari_viewer):
    """Test colormap change callback."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    lifetime_widget = parent.phasor_mapping_tab

    # Create mock event and layer
    mock_layer = MagicMock()
    mock_layer.colormap.colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    mock_layer.contrast_limits = (1.0, 5.0)

    mock_event = MagicMock()
    mock_event.source = mock_layer

    # Set initial contrast limits
    lifetime_widget.colormap_contrast_limits = (0.0, 10.0)

    with patch.object(
        lifetime_widget.histogram_widget, 'update_colormap'
    ) as mock_update_cmap:
        lifetime_widget._on_colormap_changed(mock_event)

        # Check that attributes are updated
        np.testing.assert_array_equal(
            lifetime_widget.lifetime_colormap, mock_layer.colormap.colors
        )
        assert lifetime_widget.colormap_contrast_limits == (1.0, 5.0)

        # Check that histogram colormap update is called
        mock_update_cmap.assert_called_once()

    # Test that the method skips execution when _updating_contrast_limits is True
    lifetime_widget._updating_contrast_limits = True

    with patch.object(
        lifetime_widget.histogram_widget, 'update_colormap'
    ) as mock_update_cmap:
        lifetime_widget._on_colormap_changed(mock_event)

        # Should not be called when flag is set
        mock_update_cmap.assert_not_called()

    # Reset flag
    lifetime_widget._updating_contrast_limits = False


def test_phasor_mapping_widget_calculate_lifetimes_with_real_data(
    make_napari_viewer,
):
    """Test calculating different lifetime types with real phasor data and compare with expected values."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    lifetime_widget = parent.phasor_mapping_tab

    # Set up test data
    parent.harmonic = 1
    frequency = 80.0  # MHz

    # Create realistic phasor data
    real_values = np.array([[0.5, 0.6], [0.7, 0.8]])[np.newaxis, :, :]
    imag_values = np.array([[0.3, 0.4], [0.5, 0.6]])[np.newaxis, :, :]

    layer = Image(
        np.ones((2, 2)),
        name="Test Intensity Image",
        metadata={
            "original_mean": np.ones((2, 2)),
            "settings": {},
            "G": real_values,
            "S": imag_values,
            "G_original": real_values.copy(),
            "S_original": imag_values.copy(),
            "harmonics": np.array([1]),
        },
    )

    viewer.add_layer(layer)

    # Select the layer
    parent.image_layer_with_phasor_features_combobox.setCurrentText(layer.name)
    lifetime_widget.frequency_input.setText(str(frequency))

    # Test Apparent Phase Lifetime
    lifetime_widget.lifetime_type_combobox.setCurrentText(
        "Apparent Phase Lifetime"
    )

    # Calculate expected values directly
    expected_frequency = frequency * parent.harmonic
    expected_phase_lifetime, expected_mod_lifetime = (
        phasor_to_apparent_lifetime(
            real_values, imag_values, frequency=expected_frequency
        )
    )
    expected_phase_clipped = np.clip(
        expected_phase_lifetime, a_min=0, a_max=None
    )

    # Calculate using widget
    lifetime_widget.calculate_lifetimes()

    # Compare results (widget flattens the data)
    np.testing.assert_array_almost_equal(
        lifetime_widget.lifetime_data_original,
        expected_phase_clipped.flatten(),
        decimal=10,
    )
    np.testing.assert_array_almost_equal(
        lifetime_widget.lifetime_data,
        expected_phase_clipped.flatten(),
        decimal=10,
    )

    # Test Apparent Modulation Lifetime
    lifetime_widget.lifetime_type_combobox.setCurrentText(
        "Apparent Modulation Lifetime"
    )

    # Calculate expected values
    expected_mod_clipped = np.clip(expected_mod_lifetime, a_min=0, a_max=None)

    # Calculate using widget
    lifetime_widget.calculate_lifetimes()

    # Compare results (widget flattens the data)
    np.testing.assert_array_almost_equal(
        lifetime_widget.lifetime_data_original,
        expected_mod_clipped.flatten(),
        decimal=10,
    )

    # Test Normal Lifetime
    lifetime_widget.lifetime_type_combobox.setCurrentText("Normal Lifetime")

    # Calculate expected values directly
    expected_normal_lifetime = phasor_to_normal_lifetime(
        real_values, imag_values, frequency=expected_frequency
    )

    # Calculate using widget
    lifetime_widget.calculate_lifetimes()

    # Compare results (widget flattens the data)
    np.testing.assert_array_almost_equal(
        lifetime_widget.lifetime_data_original,
        expected_normal_lifetime.flatten(),
        decimal=10,
    )


def test_phasor_mapping_widget_full_workflow_with_real_calculations(
    make_napari_viewer,
):
    """Test the complete workflow with real lifetime calculations and layer creation."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    lifetime_widget = parent.phasor_mapping_tab

    # Create and add synthetic layer
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)

    # Select the layer in the combobox
    parent.image_layer_with_phasor_features_combobox.setCurrentText(layer.name)

    # Set frequency
    lifetime_widget.frequency_input.setText("80")

    # Select lifetime type
    lifetime_widget.lifetime_type_combobox.setCurrentText(
        "Apparent Phase Lifetime"
    )

    # Trigger the calculation to create the lifetime layer
    lifetime_widget._on_calculate_lifetime_clicked()

    # Check lifetime layer was added
    assert lifetime_widget.lifetime_layer in viewer.layers

    harmonic = parent.harmonic

    # Use the new array-based metadata structure
    metadata = layer.metadata
    G_image = metadata["G"]
    S_image = metadata["S"]
    harmonics = metadata.get("harmonics", [1])

    # Handle both single harmonic and multi-harmonic cases
    harmonics = np.atleast_1d(harmonics)
    if len(harmonics) > 1 and G_image.ndim > 2:
        harmonic_idx = np.where(harmonics == harmonic)[0]
        harmonic_idx = 0 if len(harmonic_idx) == 0 else harmonic_idx[0]
        real = G_image[harmonic_idx]
        imag = S_image[harmonic_idx]
    else:
        real = G_image[0]
        imag = S_image[0]

    expected_phase_lifetime, expected_mod_lifetime = (
        phasor_to_apparent_lifetime(real, imag, frequency=80)
    )

    # Apply same clipping as the widget does
    expected_phase_lifetime = np.clip(
        expected_phase_lifetime, a_min=0, a_max=None
    )
    expected_phase_lifetime[expected_phase_lifetime < 0] = 0
    expected_mod_lifetime = np.clip(expected_mod_lifetime, a_min=0, a_max=None)
    expected_mod_lifetime[expected_mod_lifetime < 0] = 0

    lifetime_layer = viewer.layers[lifetime_widget.lifetime_layer.name]

    # Verify expected lifetime values
    np.testing.assert_allclose(
        lifetime_layer.data, expected_phase_lifetime, rtol=1e-3
    )

    # Change lifetime type to Modulation Lifetime
    lifetime_widget.lifetime_type_combobox.setCurrentText(
        "Apparent Modulation Lifetime"
    )

    # Trigger the calculation to create the lifetime layer
    lifetime_widget._on_calculate_lifetime_clicked()

    # Get the new lifetime layer (name changes when lifetime type changes)
    mod_lifetime_layer_name = f"Apparent Modulation Lifetime: {layer.name}"
    assert mod_lifetime_layer_name in viewer.layers
    mod_lifetime_layer = viewer.layers[mod_lifetime_layer_name]

    # Verify that the layer was updated with new data (not the same as phase lifetime)
    assert not np.array_equal(mod_lifetime_layer.data, expected_phase_lifetime)
    # Verify layer name reflects the new lifetime type
    assert "Apparent Modulation Lifetime" in mod_lifetime_layer.name

    # Change lifetime type to Normal Lifetime
    lifetime_widget.lifetime_type_combobox.setCurrentText("Normal Lifetime")

    # Trigger the calculation to create the lifetime layer
    lifetime_widget._on_calculate_lifetime_clicked()

    # Get the new lifetime layer (name changes when lifetime type changes)
    normal_lifetime_layer_name = f"Normal Lifetime: {layer.name}"
    assert normal_lifetime_layer_name in viewer.layers
    normal_lifetime_layer = viewer.layers[normal_lifetime_layer_name]

    # Verify that the layer was updated again with different data
    assert not np.array_equal(
        normal_lifetime_layer.data, expected_phase_lifetime
    )
    assert not np.array_equal(
        normal_lifetime_layer.data, mod_lifetime_layer.data
    )
    # Verify layer name reflects the new lifetime type
    assert "Normal Lifetime" in normal_lifetime_layer.name


def test_phasor_mapping_widget_range_clipping_with_real_data(
    make_napari_viewer,
):
    """Test range clipping functionality with real calculated lifetime data and slider interaction."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    lifetime_widget = parent.phasor_mapping_tab

    # Create and add real layer with phasor data
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)

    # Set up test parameters
    frequency = 80.0
    lifetime_widget.frequency_input.setText(str(frequency))

    # Set normal lifetime type and calculate
    lifetime_widget.lifetime_type_combobox.setCurrentText("Normal Lifetime")

    # Trigger the calculation to create the lifetime layer
    lifetime_widget._on_calculate_lifetime_clicked()

    # Verify layer was created and data calculated
    assert lifetime_widget.lifetime_layer in viewer.layers
    assert lifetime_widget.lifetime_data_original is not None
    assert lifetime_widget.lifetime_data is not None

    # Get the original calculated data
    original_data = lifetime_widget.lifetime_data_original.copy()

    # Verify initial state - data should be unclipped
    np.testing.assert_allclose(
        lifetime_widget.lifetime_data, original_data, rtol=1e-3
    )

    # Test range clipping with slider
    # Get the actual lifetime range from calculated data
    valid_lifetimes = original_data[np.isfinite(original_data)]
    min_lifetime = np.min(valid_lifetimes)
    max_lifetime = np.max(valid_lifetimes)
    lifetime_range = max_lifetime - min_lifetime

    # Define clipping range - clip to middle 60% of the data range
    clip_min = min_lifetime + 0.2 * lifetime_range
    clip_max = max_lifetime - 0.2 * lifetime_range

    # Convert to slider values (multiply by lifetime_range_factor)
    min_slider = int(clip_min * lifetime_widget.lifetime_range_factor)
    max_slider = int(clip_max * lifetime_widget.lifetime_range_factor)

    # Apply clipping via slider change
    lifetime_widget.lifetime_range_slider.setValue((min_slider, max_slider))
    lifetime_widget._on_lifetime_range_changed((min_slider, max_slider))

    # Calculate expected clipped data
    expected_clipped = np.clip(original_data, clip_min, clip_max)

    # Verify clipping worked correctly on widget data
    np.testing.assert_array_almost_equal(
        lifetime_widget.lifetime_data, expected_clipped, decimal=3
    )

    # Verify the layer was updated with clipped data
    # Layer data is in 2D shape, so compare flattened version
    assert lifetime_widget.lifetime_layer is not None
    np.testing.assert_array_almost_equal(
        lifetime_widget.lifetime_layer.data.flatten(),
        expected_clipped,
        decimal=3,
    )

    # Verify contrast limits were updated
    assert lifetime_widget.lifetime_layer is not None
    contrast_limits = lifetime_widget.lifetime_layer.contrast_limits
    assert len(contrast_limits) == 2
    assert abs(contrast_limits[0] - clip_min) < 0.01
    assert abs(contrast_limits[1] - clip_max) < 0.01

    # Test slider UI updates
    assert lifetime_widget.lifetime_range_slider.value() == (
        min_slider,
        max_slider,
    )
    assert (
        abs(float(lifetime_widget.lifetime_min_edit.text()) - clip_min) < 0.01
    )
    assert (
        abs(float(lifetime_widget.lifetime_max_edit.text()) - clip_max) < 0.01
    )

    # Test more aggressive clipping - clip to middle 20% of range
    clip_min_tight = min_lifetime + 0.4 * lifetime_range
    clip_max_tight = max_lifetime - 0.4 * lifetime_range

    min_slider_tight = int(
        clip_min_tight * lifetime_widget.lifetime_range_factor
    )
    max_slider_tight = int(
        clip_max_tight * lifetime_widget.lifetime_range_factor
    )

    # Apply tighter clipping
    lifetime_widget.lifetime_range_slider.setValue(
        (min_slider_tight, max_slider_tight)
    )
    lifetime_widget._on_lifetime_range_changed(
        (min_slider_tight, max_slider_tight)
    )

    expected_clipped_tight = np.clip(
        original_data, clip_min_tight, clip_max_tight
    )

    # Verify tighter clipping
    np.testing.assert_array_almost_equal(
        lifetime_widget.lifetime_data, expected_clipped_tight, decimal=3
    )

    np.testing.assert_array_almost_equal(
        lifetime_widget.lifetime_layer.data.flatten(),
        expected_clipped_tight,
        decimal=3,
    )

    # Verify contrast limits for tighter clipping
    contrast_limits_tight = lifetime_widget.lifetime_layer.contrast_limits
    assert abs(contrast_limits_tight[0] - clip_min_tight) < 0.01
    assert abs(contrast_limits_tight[1] - clip_max_tight) < 0.01

    # Test resetting to full range
    full_min_slider = int(min_lifetime * lifetime_widget.lifetime_range_factor)
    full_max_slider = int(max_lifetime * lifetime_widget.lifetime_range_factor)

    lifetime_widget.lifetime_range_slider.setValue(
        (full_min_slider, full_max_slider)
    )
    lifetime_widget._on_lifetime_range_changed(
        (full_min_slider, full_max_slider)
    )

    # Verify data is back to original (unclipped) state
    np.testing.assert_array_almost_equal(
        lifetime_widget.lifetime_data, original_data, decimal=3
    )

    # Verify contrast limits are reset to full range
    contrast_limits_full = lifetime_widget.lifetime_layer.contrast_limits
    assert abs(contrast_limits_full[0] - min_lifetime) < 0.01
    assert abs(contrast_limits_full[1] - max_lifetime) < 0.01

    # Test slider drag state during range changes
    assert lifetime_widget.histogram_widget._slider_being_dragged is False

    # Simulate slider being dragged
    lifetime_widget.histogram_widget._on_slider_pressed()
    assert lifetime_widget.histogram_widget._slider_being_dragged is True

    # Change range while dragging
    lifetime_widget.lifetime_range_slider.setValue((min_slider, max_slider))
    lifetime_widget._on_lifetime_range_changed((min_slider, max_slider))

    # Verify data still updated even while dragging
    np.testing.assert_array_almost_equal(
        lifetime_widget.lifetime_data, expected_clipped, decimal=3
    )
    np.testing.assert_array_almost_equal(
        lifetime_widget.lifetime_layer.data.flatten(),
        expected_clipped,
        decimal=3,
    )

    # Release slider
    lifetime_widget.histogram_widget._on_slider_released()
    assert lifetime_widget.histogram_widget._slider_being_dragged is False

    # Test histogram update after clipping
    with patch.object(
        lifetime_widget.histogram_widget, 'update_data'
    ) as mock_update_data:
        mock_update_data.reset_mock()  # Reset any previous calls
        lifetime_widget.lifetime_range_slider.setValue(
            (min_slider, max_slider)
        )
        lifetime_widget._on_lifetime_range_changed((min_slider, max_slider))
        mock_update_data.assert_called_once()


def test_phasor_mapping_widget_different_harmonics_and_frequencies(
    make_napari_viewer,
):
    """Test lifetime calculations with different harmonic and frequency combinations."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    lifetime_widget = parent.phasor_mapping_tab

    # Test data - 2x2 arrays
    real_values = np.array([[0.6, 0.7], [0.8, 0.5]])[np.newaxis, :, :]
    imag_values = np.array([[0.4, 0.3], [0.2, 0.5]])[np.newaxis, :, :]

    # Test different combinations
    test_cases = [
        (1, 80.0),  # 1st harmonic, 80 MHz
        (2, 80.0),  # 2nd harmonic, 80 MHz
        (1, 40.0),  # 1st harmonic, 40 MHz
        (3, 160.0),  # 3rd harmonic, 160 MHz
    ]

    for harmonic, base_frequency in test_cases:
        layer = Image(
            np.ones((2, 2)),
            name="Test Intensity Image",
            metadata={
                "original_mean": np.ones((2, 2)),
                "settings": {},
                "G": real_values,
                "S": imag_values,
                "G_original": real_values.copy(),
                "S_original": imag_values.copy(),
                "harmonics": np.array([harmonic]),
            },
        )

        viewer.add_layer(layer)

        # Set up for this test case
        parent.harmonic = harmonic

        # Select the layer in the combobox
        parent.image_layer_with_phasor_features_combobox.setCurrentText(
            layer.name
        )

        lifetime_widget.frequency_input.setText(str(base_frequency))

        # Calculate expected values
        expected_frequency = base_frequency * harmonic
        expected_phase_lifetime, _ = phasor_to_apparent_lifetime(
            real_values, imag_values, frequency=expected_frequency
        )
        expected_clipped = np.clip(
            expected_phase_lifetime, a_min=0, a_max=None
        )

        # Calculate using widget
        lifetime_widget.lifetime_type_combobox.setCurrentText(
            "Apparent Phase Lifetime"
        )
        lifetime_widget.calculate_lifetimes()

        # Verify results for this combination (widget flattens the data)
        np.testing.assert_array_almost_equal(
            lifetime_widget.lifetime_data_original,
            expected_clipped.flatten(),
            decimal=10,
            err_msg=f"Failed for harmonic={harmonic}, frequency={base_frequency}",
        )

        # Verify frequency was calculated correctly
        assert lifetime_widget.frequency == base_frequency

        # Clean up layer for next iteration
        viewer.layers.remove(layer)


def test_phasor_mapping_widget_output_mode_updates_button_text(
    make_napari_viewer,
):
    """Button text should reflect the selected mapping parameter."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    mapping_widget = parent.phasor_mapping_tab

    assert (
        mapping_widget.calculate_lifetime_button.text()
        == "Display Lifetime Map"
    )

    mapping_widget.output_mode_combobox.setCurrentText("Phase")
    assert (
        mapping_widget.calculate_lifetime_button.text() == "Display Phase Map"
    )

    mapping_widget.output_mode_combobox.setCurrentText("Modulation")
    assert (
        mapping_widget.calculate_lifetime_button.text()
        == "Display Modulation Map"
    )


def test_phasor_mapping_widget_apply_2d_text_tracks_plot_type(
    make_napari_viewer,
):
    """Checkbox text should follow the active plot artist type."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    mapping_widget = parent.phasor_mapping_tab

    assert (
        mapping_widget.apply_2d_colormap_checkbox.text()
        == "Apply colormap to 2D Histogram"
    )

    parent.plotter_inputs_widget.plot_type_combobox.setCurrentText(
        "Dot Plot (Scatter)"
    )
    assert (
        mapping_widget.apply_2d_colormap_checkbox.text()
        == "Apply colormap to Scatter plot"
    )

    parent.plotter_inputs_widget.plot_type_combobox.setCurrentText(
        "Contour Plot"
    )
    assert (
        mapping_widget.apply_2d_colormap_checkbox.text()
        == "Apply colormap to Contour plot"
    )

    parent.plotter_inputs_widget.plot_type_combobox.setCurrentText("None")
    assert (
        mapping_widget.apply_2d_colormap_checkbox.text()
        == "Apply colormap to Plot"
    )

    parent.plotter_inputs_widget.plot_type_combobox.setCurrentText(
        "Density Plot (2D Histogram)"
    )
    assert (
        mapping_widget.apply_2d_colormap_checkbox.text()
        == "Apply colormap to 2D Histogram"
    )


def test_phasor_mapping_widget_phase_modulation_calculation(
    make_napari_viewer,
):
    """Phase and modulation outputs should match phasor_to_polar values."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    mapping_widget = parent.phasor_mapping_tab

    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)
    parent.image_layer_with_phasor_features_combobox.setCurrentText(layer.name)

    metadata = layer.metadata
    g_array = metadata["G"]
    s_array = metadata["S"]
    harmonics = np.atleast_1d(metadata.get("harmonics", [parent.harmonic]))
    harmonic_index = int(np.where(harmonics == parent.harmonic)[0][0])
    real = g_array[harmonic_index]
    imag = s_array[harmonic_index]
    expected_phase, expected_modulation = phasor_to_polar(real, imag)

    mapping_widget.output_mode_combobox.setCurrentText("Phase")
    mapping_widget.calculate_output_data()
    np.testing.assert_allclose(
        mapping_widget.current_metric_data_original,
        expected_phase.flatten(),
        rtol=1e-6,
        atol=1e-6,
    )

    mapping_widget.output_mode_combobox.setCurrentText("Modulation")
    mapping_widget.calculate_output_data()
    np.testing.assert_allclose(
        mapping_widget.current_metric_data_original,
        expected_modulation.flatten(),
        rtol=1e-6,
        atol=1e-6,
    )

    assert "derived_data" in layer.metadata
    assert "Phase" in layer.metadata["derived_data"]
    assert "Modulation" in layer.metadata["derived_data"]


def test_phasor_mapping_widget_phase_modulation_layer_display(
    make_napari_viewer,
):
    """Display action should create/update phase and modulation map layers."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    mapping_widget = parent.phasor_mapping_tab

    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)
    parent.image_layer_with_phasor_features_combobox.setCurrentText(layer.name)

    mapping_widget.output_mode_combobox.setCurrentText("Phase")
    mapping_widget.colormap_combobox.setCurrentText("viridis")
    mapping_widget._on_calculate_lifetime_clicked()

    phase_layer_name = f"Phase: {layer.name}"
    assert phase_layer_name in viewer.layers
    phase_layer = viewer.layers[phase_layer_name]
    assert phase_layer.colormap.name == "viridis"

    mapping_widget.output_mode_combobox.setCurrentText("Modulation")
    mapping_widget.colormap_combobox.setCurrentText("plasma")
    mapping_widget._on_calculate_lifetime_clicked()

    modulation_layer_name = f"Modulation: {layer.name}"
    assert modulation_layer_name in viewer.layers
    modulation_layer = viewer.layers[modulation_layer_name]
    assert modulation_layer.colormap.name == "plasma"


def test_phasor_mapping_widget_select_color_uses_napari_colormap(
    make_napari_viewer,
):
    """Sentinel colormap entry should resolve to a real napari colormap."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    mapping_widget = parent.phasor_mapping_tab

    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)
    parent.image_layer_with_phasor_features_combobox.setCurrentText(layer.name)

    mapping_widget.output_mode_combobox.setCurrentText("Phase")
    mapping_widget._set_custom_color(QColor(12, 34, 56))
    mapping_widget.colormap_combobox.setCurrentText("Select color...")
    mapping_widget._on_calculate_lifetime_clicked()

    phase_layer_name = f"Phase: {layer.name}"
    assert phase_layer_name in viewer.layers
    phase_layer = viewer.layers[phase_layer_name]

    assert hasattr(phase_layer.colormap, "colors")
    assert phase_layer.colormap.name != "Select color..."
    expected = np.array([12 / 255, 34 / 255, 56 / 255], dtype=np.float32)
    np.testing.assert_allclose(
        phase_layer.colormap.colors[-1][:3], expected, atol=1e-3
    )
    np.testing.assert_allclose(
        phase_layer.colormap.colors[0][:3], np.zeros(3), atol=1e-6
    )

    mapping_widget.apply_2d_colormap_checkbox.setChecked(True)
    mapping_widget._set_custom_color(QColor(100, 50, 25))
    mapping_widget._on_colormap_combobox_changed("Select color...")

    updated_expected = np.array(
        [100 / 255, 50 / 255, 25 / 255], dtype=np.float32
    )
    np.testing.assert_allclose(
        phase_layer.colormap.colors[-1][:3], updated_expected, atol=1e-3
    )
    assert phase_layer.colormap.name != "Select color..."


def test_phasor_mapping_histogram_overlay_checkbox_lifecycle(
    make_napari_viewer,
):
    """Phase/Modulation overlay should be created and cleared via checkbox."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    mapping_widget = parent.phasor_mapping_tab

    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)
    parent.image_layer_with_phasor_features_combobox.setCurrentText(layer.name)
    parent.on_image_layer_changed()

    # Start with Phase mode and no overlay.
    mapping_widget.output_mode_combobox.setCurrentText("Phase")
    mapping_widget._on_calculate_lifetime_clicked()
    assert mapping_widget._overlay_imshow is None

    hist_artist = parent.canvas_widget.artists["HISTOGRAM2D"]
    histogram_img = hist_artist._mpl_artists.get("histogram_image")
    assert histogram_img is not None
    assert histogram_img.get_visible()

    # Enabling 2D colormap should create overlay and hide base density image.
    mapping_widget.apply_2d_colormap_checkbox.setChecked(True)
    assert mapping_widget._overlay_imshow is not None
    assert not histogram_img.get_visible()

    # Switch output type to Modulation: overlay should remain active.
    mapping_widget.output_mode_combobox.setCurrentText("Modulation")
    mapping_widget._on_calculate_lifetime_clicked()
    assert mapping_widget._overlay_imshow is not None
    assert not histogram_img.get_visible()

    # Turning the checkbox off should clear overlay and show density image.
    mapping_widget.apply_2d_colormap_checkbox.setChecked(False)
    assert mapping_widget._overlay_imshow is None
    assert histogram_img.get_visible()


def test_phasor_mapping_histogram_overlay_tab_visibility_lifecycle(
    make_napari_viewer,
):
    """Overlay should clear when tab is hidden and reapply when shown again."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    mapping_widget = parent.phasor_mapping_tab

    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)
    parent.image_layer_with_phasor_features_combobox.setCurrentText(layer.name)
    parent.on_image_layer_changed()

    mapping_widget.output_mode_combobox.setCurrentText("Phase")
    mapping_widget._on_calculate_lifetime_clicked()
    mapping_widget.apply_2d_colormap_checkbox.setChecked(True)
    assert mapping_widget._overlay_imshow is not None

    hist_artist = parent.canvas_widget.artists["HISTOGRAM2D"]
    histogram_img = hist_artist._mpl_artists.get("histogram_image")
    assert histogram_img is not None
    assert not histogram_img.get_visible()

    # Simulate leaving the tab: overlay should be removed and base density restored.
    mapping_widget.on_tab_visibility_changed(False)
    assert mapping_widget._overlay_imshow is None
    assert histogram_img.get_visible()

    # Simulate returning to tab: overlay should be reapplied.
    mapping_widget.on_tab_visibility_changed(True)
    assert mapping_widget._overlay_imshow is not None
    assert not histogram_img.get_visible()


def test_mesh_overlay_independent_from_apply_colormap_toggle(
    make_napari_viewer,
):
    """Mesh should remain visible when colormap toggle is off."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    mapping_widget = parent.phasor_mapping_tab

    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)
    parent.image_layer_with_phasor_features_combobox.setCurrentText(layer.name)
    parent.on_image_layer_changed()

    mapping_widget.output_mode_combobox.setCurrentText("Phase")
    mapping_widget._on_calculate_lifetime_clicked()

    hist_artist = parent.canvas_widget.artists["HISTOGRAM2D"]
    histogram_img = hist_artist._mpl_artists.get("histogram_image")
    assert histogram_img is not None

    mapping_widget.apply_2d_colormap_checkbox.setChecked(False)
    mapping_widget.mesh_overlay_checkbox.setChecked(True)

    assert mapping_widget._overlay_imshow is not None
    # With independent toggles, density image remains visible when
    # apply-colormap is off even if mesh is on.
    assert histogram_img.get_visible()


def test_mesh_settings_persist_across_layer_switches(make_napari_viewer):
    """Mesh toggle, alpha, and ranges should restore per layer."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    mapping_widget = parent.phasor_mapping_tab

    layer_1 = create_image_layer_with_phasors()
    layer_2 = create_image_layer_with_phasors()
    viewer.add_layer(layer_1)
    viewer.add_layer(layer_2)

    parent.image_layer_with_phasor_features_combobox.setCurrentText(
        layer_1.name
    )
    parent.on_image_layer_changed()

    mapping_widget.output_mode_combobox.setCurrentText("Phase")
    mapping_widget._on_calculate_lifetime_clicked()

    mapping_widget.mesh_overlay_checkbox.setChecked(True)
    mapping_widget.mesh_alpha_spinbox.setValue(0.62)
    mapping_widget.phase_range_slider.setValue((25, 120))
    mapping_widget.modulation_range_slider.setValue((10, 70))

    parent.image_layer_with_phasor_features_combobox.setCurrentText(
        layer_2.name
    )
    parent.on_image_layer_changed()

    parent.image_layer_with_phasor_features_combobox.setCurrentText(
        layer_1.name
    )
    parent.on_image_layer_changed()

    assert mapping_widget.mesh_overlay_checkbox.isChecked()
    assert abs(mapping_widget.mesh_alpha_spinbox.value() - 0.62) < 1e-6
    assert mapping_widget.phase_range_slider.value() == (25, 120)
    assert mapping_widget.modulation_range_slider.value() == (10, 70)


def test_full_circle_mesh_supports_phase_over_pi(make_napari_viewer):
    """Full-circle mode should keep mesh values for phases > pi."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    mapping_widget = parent.phasor_mapping_tab

    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)
    parent.image_layer_with_phasor_features_combobox.setCurrentText(layer.name)
    parent.on_image_layer_changed()

    mapping_widget.output_mode_combobox.setCurrentText("Phase")
    mapping_widget._on_calculate_lifetime_clicked()

    # Full-circle toggle ON means full circle mode.
    parent.plotter_inputs_widget.semi_circle_checkbox.setChecked(True)

    mapping_widget.mesh_overlay_checkbox.setChecked(True)
    lower = int(3.5 * mapping_widget.phase_range_factor)
    upper = int(4.2 * mapping_widget.phase_range_factor)
    mapping_widget.phase_range_slider.setValue((lower, upper))

    assert mapping_widget._overlay_imshow is not None
    arr = np.asarray(mapping_widget._overlay_imshow.get_array())
    assert np.isfinite(arr).any()


def test_mesh_redraw_is_debounced_on_axes_limit_changes(make_napari_viewer):
    """Axes changes should schedule one deferred mesh redraw via timer."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    mapping_widget = parent.phasor_mapping_tab

    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)
    parent.image_layer_with_phasor_features_combobox.setCurrentText(layer.name)
    parent.on_image_layer_changed()

    mapping_widget.output_mode_combobox.setCurrentText("Phase")
    mapping_widget._on_calculate_lifetime_clicked()
    mapping_widget.mesh_overlay_checkbox.setChecked(True)
    mapping_widget._coloring_paused_by_tab = False

    with patch.object(
        mapping_widget, '_apply_histogram_coloring'
    ) as mock_apply:
        mapping_widget._on_axes_limits_changed(parent.canvas_widget.axes)
        assert mapping_widget._mesh_axes_update_timer.isActive()
        mock_apply.assert_not_called()

        mapping_widget._apply_mesh_after_axes_change()
        mock_apply.assert_called_once_with("Phase")
