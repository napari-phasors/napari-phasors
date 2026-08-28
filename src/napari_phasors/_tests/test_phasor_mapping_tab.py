from unittest.mock import MagicMock, patch

import matplotlib.colors as mcolors
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
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
from napari_phasors.phasor_mapping_tab import (
    _DEFAULT_MESH_RESOLUTION,
    PhasorMappingWidget,
    _resolve_mesh_blur_sigma,
    draw_phasor_mesh,
)
from napari_phasors.plotter import PlotterWidget


def test_phasor_mapping_widget_initialization_values(make_viewer_model, qtbot):
    """Test the initialization of the Lifetime Widget."""
    viewer = make_viewer_model()
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
        lifetime_widget.lifetime_range_label.text() == "Lifetime range (ns):"
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
    # The horizontal scrollbar must only appear once the content genuinely
    # can't shrink further, not be permanently suppressed.
    assert scroll_area.horizontalScrollBarPolicy() == Qt.ScrollBarAsNeeded

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


def test_phasor_mapping_widget_histogram_styling(make_viewer_model, qtbot):
    """Test that histogram styling is applied correctly."""
    viewer = make_viewer_model()

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


def test_mesh_alpha_map_is_cached_for_repeated_refreshes(
    make_viewer_model, qtbot
):
    """Test that the blurred mesh alpha map is reused for identical mesh state."""
    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    lifetime_widget = parent.phasor_mapping_tab

    mesh_mask = np.array([[False, True], [True, False]])
    alpha_key = (1, 2, 3, 4, True, 320, 0, 628, 0, 100, False)
    blurred_base = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=float)

    with patch(
        'napari_phasors.phasor_mapping_tab.gaussian_filter',
        return_value=blurred_base,
    ) as mock_filter:
        ax = parent.canvas_widget.axes
        first = lifetime_widget._get_mesh_alpha_map(
            mesh_mask, alpha_key, 0.5, 300, ax
        )
        second = lifetime_widget._get_mesh_alpha_map(
            mesh_mask, alpha_key, 0.25, 300, ax
        )

    assert mock_filter.call_count == 1
    np.testing.assert_allclose(first, blurred_base * 0.5)
    np.testing.assert_allclose(second, blurred_base * 0.25)


def test_phasor_mapping_widget_frequency_input_validation(
    make_viewer_model, qtbot
):
    """Test frequency input validation."""
    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    lifetime_widget = parent.phasor_mapping_tab

    # Test that frequency input has double validator
    validator = lifetime_widget.frequency_input.validator()
    assert validator is not None

    # Test valid input
    lifetime_widget.frequency_input.setText("80.0")
    assert lifetime_widget.frequency_input.text() == "80.0"


def test_phasor_mapping_widget_slider_drag_state(make_viewer_model, qtbot):
    """Test slider drag state tracking."""
    viewer = make_viewer_model()
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


def test_phasor_mapping_widget_range_label_update(make_viewer_model, qtbot):
    """Test lifetime range label update."""
    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    lifetime_widget = parent.phasor_mapping_tab

    # Test edits update while dragging (label keeps just the prefix)
    test_value = (25000, 75000)  # Represents 25.0 - 75.0 ns with factor 1000
    lifetime_widget.histogram_widget._on_range_label_update(test_value)

    assert (
        lifetime_widget.lifetime_range_label.text() == "Lifetime range (ns):"
    )
    assert lifetime_widget.lifetime_min_edit.text() == "25.00"
    assert lifetime_widget.lifetime_max_edit.text() == "75.00"


def test_phasor_mapping_widget_calculate_lifetimes_no_layer(
    make_viewer_model,
    qtbot,
):
    """Test calculate_lifetimes when no layer is available."""
    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    lifetime_widget = parent.phasor_mapping_tab

    # Should return early without error
    lifetime_widget.calculate_lifetimes()
    assert lifetime_widget.lifetime_data_original is None


def test_phasor_mapping_widget_plot_histogram_no_data(
    make_viewer_model, qtbot
):
    """Test plotting histogram when no data is available."""
    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    lifetime_widget = parent.phasor_mapping_tab
    parent.tab_widget.setCurrentWidget(lifetime_widget)

    # No data should leave the histogram empty but VISIBLE with controls disabled.
    lifetime_widget.plot_lifetime_histogram()
    assert lifetime_widget.histogram_widget.counts is None
    assert not lifetime_widget.histogram_widget.isHidden()
    assert not lifetime_widget.histogram_widget._settings_button.isEnabled()
    assert not lifetime_widget.histogram_widget.save_button.isEnabled()


def test_phasor_mapping_widget_ui_layout(make_viewer_model, qtbot):
    """Test the UI layout structure."""
    viewer = make_viewer_model()
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


def test_phasor_mapping_widget_canvas_properties(make_viewer_model, qtbot):
    """Test canvas and figure properties."""
    viewer = make_viewer_model()
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
    assert canvas.height() == 180  # Minimum canvas height set in the widget


def test_phasor_mapping_widget_type_changed_no_frequency(
    make_viewer_model, qtbot
):
    """Test behavior when Calculate is clicked but no frequency is set."""
    viewer = make_viewer_model()
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
    make_viewer_model,
    qtbot,
):
    """Test that lifetime settings are only initialized when analysis is performed."""
    viewer = make_viewer_model()
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


def test_phasor_mapping_widget_settings_update_in_metadata(
    make_viewer_model, qtbot
):
    """Test that changing settings updates layer metadata."""
    viewer = make_viewer_model()
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
    make_viewer_model,
    qtbot,
):
    """Test that settings persist when switching between layers."""
    viewer = make_viewer_model()
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
    make_viewer_model,
    qtbot,
):
    """Test that adding a layer without lifetime settings doesn't auto-initialize metadata."""
    viewer = make_viewer_model()
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
    make_viewer_model,
    qtbot,
):
    """Test that lifetime range settings are restored after recalculating lifetimes."""
    viewer = make_viewer_model()
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
    make_viewer_model,
    qtbot,
):
    """Test that adding/removing layers properly manages settings."""
    viewer = make_viewer_model()
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
    make_viewer_model,
    qtbot,
):
    """Test that frequency is saved to metadata when Calculate is clicked."""
    viewer = make_viewer_model()
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
    make_viewer_model,
    qtbot,
):
    """Test that restoring settings doesn't trigger recursive updates."""
    viewer = make_viewer_model()
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


def test_phasor_mapping_widget_slider_range_update(make_viewer_model, qtbot):
    """Test updating slider range based on lifetime data."""
    viewer = make_viewer_model()
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
    make_viewer_model,
    qtbot,
):
    """Test updating slider range when no valid data exists."""
    viewer = make_viewer_model()
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


def test_phasor_mapping_widget_min_max_edit_callbacks(
    make_viewer_model, qtbot
):
    """Test manual entry of min/max values."""
    viewer = make_viewer_model()
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
    make_viewer_model,
    qtbot,
):
    """Test behavior when image layer changes and has frequency settings."""
    viewer = make_viewer_model()
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
    make_viewer_model,
    qtbot,
):
    """Test behavior when no layer is selected."""
    viewer = make_viewer_model()
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
    assert not lifetime_widget.histogram_widget.save_button.isEnabled()


def test_phasor_mapping_widget_colormap_changed_callback(
    make_viewer_model, qtbot
):
    """Test colormap change callback."""
    viewer = make_viewer_model()
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


def test_phasor_mapping_gamma_links_layers_and_histogram(
    make_viewer_model,
    qtbot,
):
    """Changing gamma on one lifetime layer syncs siblings and the histogram."""
    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    lifetime_widget = parent.phasor_mapping_tab

    layer_1 = create_image_layer_with_phasors()
    layer_2 = create_image_layer_with_phasors()
    viewer.add_layer(layer_1)
    viewer.add_layer(layer_2)

    lifetime_widget.frequency_input.setText("80.0")
    parent._broadcast_frequency_value_across_tabs("80.0")
    lifetime_widget.lifetime_type_combobox.setCurrentText("Normal Lifetime")

    with patch.object(
        parent, "get_selected_layers", return_value=[layer_1, layer_2]
    ):
        lifetime_widget._on_calculate_lifetime_clicked()

    assert len(lifetime_widget.metric_layers) == 2

    # Changing gamma on one output layer propagates to the sibling layer, the
    # stored gamma, and the histogram widget.
    lifetime_widget.metric_layers[0].gamma = 0.6

    assert lifetime_widget.metric_layers[1].gamma == 0.6
    assert lifetime_widget.colormap_gamma == 0.6
    assert lifetime_widget.histogram_widget.gamma == 0.6


def test_phasor_mapping_widget_calculate_lifetimes_with_real_data(
    make_viewer_model,
    qtbot,
):
    """Test calculating different lifetime types with real phasor data and compare with expected values."""
    viewer = make_viewer_model()
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
    make_viewer_model,
    qtbot,
):
    """Test the complete workflow with real lifetime calculations and layer creation."""
    viewer = make_viewer_model()
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
    make_viewer_model,
    qtbot,
):
    """Test range clipping functionality with real calculated lifetime data and slider interaction."""
    viewer = make_viewer_model()
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
    make_viewer_model,
    qtbot,
):
    """Test lifetime calculations with different harmonic and frequency combinations."""
    viewer = make_viewer_model()
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
    make_viewer_model,
    qtbot,
):
    """Button text should reflect the selected mapping parameter."""
    viewer = make_viewer_model()
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
    make_viewer_model,
    qtbot,
):
    """Checkbox text should follow the active plot artist type."""
    viewer = make_viewer_model()
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
    make_viewer_model,
    qtbot,
):
    """Phase and modulation outputs should match phasor_to_polar values."""
    viewer = make_viewer_model()
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
    make_viewer_model,
    qtbot,
):
    """Display action should create/update phase and modulation map layers."""
    viewer = make_viewer_model()
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


def test_phase_output_wraps_to_full_circle_in_full_polar_mode(
    make_viewer_model,
    qtbot,
):
    """Phase output should be wrapped to [0, 2pi] in full polar mode."""
    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    mapping_widget = parent.phasor_mapping_tab

    # Build a minimal layer with known negative phase values from phasor_to_polar.
    # S < 0 yields negative angles that must wrap in full polar mode.
    real = np.array([[[0.5, -0.5], [0.25, -0.25]]], dtype=float)
    imag = np.array([[[-0.5, -0.5], [-0.25, -0.25]]], dtype=float)
    layer = Image(
        np.ones((2, 2), dtype=float),
        name="PhaseWrapLayer",
        metadata={
            "original_mean": np.ones((2, 2), dtype=float),
            "settings": {},
            "G": real,
            "S": imag,
            "G_original": real.copy(),
            "S_original": imag.copy(),
            "harmonics": np.array([1]),
        },
    )
    viewer.add_layer(layer)
    parent.image_layer_with_phasor_features_combobox.setCurrentText(layer.name)

    expected_phase, _ = phasor_to_polar(real[0], imag[0])
    expected_wrapped = np.mod(expected_phase, 2.0 * np.pi).flatten()

    # Full Polar Plot toggle ON means full-circle mode.
    parent.plotter_inputs_widget.semi_circle_checkbox.setChecked(True)
    mapping_widget.output_mode_combobox.setCurrentText("Phase")
    mapping_widget.calculate_output_data()

    np.testing.assert_allclose(
        mapping_widget.current_metric_data_original,
        expected_wrapped,
        rtol=1e-6,
        atol=1e-6,
    )
    assert np.all(mapping_widget.current_metric_data_original >= 0.0)
    assert np.all(mapping_widget.current_metric_data_original <= 2.0 * np.pi)


def test_phase_range_defaults_to_0_2pi_in_full_polar_mode(
    make_viewer_model,
    qtbot,
):
    """Phase map slider range should initialize to 0..2pi in full polar mode."""
    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    mapping_widget = parent.phasor_mapping_tab

    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)
    parent.image_layer_with_phasor_features_combobox.setCurrentText(layer.name)

    # Full Polar Plot toggle ON means full-circle mode.
    parent.plotter_inputs_widget.semi_circle_checkbox.setChecked(True)
    mapping_widget.output_mode_combobox.setCurrentText("Phase")
    mapping_widget._on_calculate_lifetime_clicked()

    expected_max = int(2.0 * np.pi * mapping_widget.lifetime_range_factor)
    min_slider, max_slider = mapping_widget.lifetime_range_slider.value()

    assert mapping_widget.lifetime_range_slider.minimum() == 0
    assert mapping_widget.lifetime_range_slider.maximum() == expected_max
    assert min_slider == 0
    assert max_slider == expected_max
    assert abs(mapping_widget.min_lifetime - 0.0) < 1e-9
    assert abs(mapping_widget.max_lifetime - (2.0 * np.pi)) < 1e-6


def test_phasor_mapping_widget_select_color_uses_napari_colormap(
    make_viewer_model,
    qtbot,
):
    """Sentinel colormap entry should resolve to a real napari colormap."""
    viewer = make_viewer_model()
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
    make_viewer_model,
    qtbot,
):
    """Phase/Modulation overlay should be created and cleared via checkbox."""
    viewer = make_viewer_model()
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
    make_viewer_model,
    qtbot,
):
    """Overlay should clear when tab is hidden and reapply when shown again."""
    viewer = make_viewer_model()
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
    make_viewer_model,
    qtbot,
):
    """Mesh should remain visible when colormap toggle is off."""
    viewer = make_viewer_model()
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

    assert mapping_widget._mesh_overlay_imshow is not None
    assert mapping_widget._overlay_imshow is None
    # With independent toggles, density image remains visible when
    # apply-colormap is off even if mesh is on.
    assert histogram_img.get_visible()

    # Turning on apply-colormap should add the plot overlay without
    # removing mesh overlay.
    mapping_widget.apply_2d_colormap_checkbox.setChecked(True)
    assert mapping_widget._mesh_overlay_imshow is not None
    assert mapping_widget._overlay_imshow is not None
    assert not histogram_img.get_visible()


def test_mesh_settings_persist_across_layer_switches(make_viewer_model, qtbot):
    """Mesh toggle, alpha, and ranges should restore per layer."""
    viewer = make_viewer_model()
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


def test_full_circle_mesh_supports_phase_over_pi(make_viewer_model, qtbot):
    """Full-circle mode should keep mesh values for phases > pi."""
    viewer = make_viewer_model()
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

    assert mapping_widget._mesh_overlay_imshow is not None
    arr = np.asarray(mapping_widget._mesh_overlay_imshow.get_array())
    assert np.isfinite(arr).any()


def test_mesh_redraw_is_debounced_on_axes_limit_changes(
    make_viewer_model, qtbot
):
    """Axes changes should schedule one deferred mesh redraw via timer."""
    viewer = make_viewer_model()
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


def test_mesh_overlay_colorbar_and_alpha_updates(make_viewer_model, qtbot):
    """Mesh colorbar toggle and alpha updates should reflect dynamically."""
    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    mapping_widget = parent.phasor_mapping_tab

    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)
    parent.image_layer_with_phasor_features_combobox.setCurrentText(layer.name)
    parent.on_image_layer_changed()

    mapping_widget.output_mode_combobox.setCurrentText("Phase")
    mapping_widget._on_calculate_lifetime_clicked()

    # Turn mesh on
    mapping_widget.mesh_overlay_checkbox.setChecked(True)
    assert not mapping_widget.mesh_colorbar_checkbox.isHidden()

    # Enable colorbar
    mapping_widget.mesh_colorbar_checkbox.setChecked(True)
    # The parent plotter coordinates the actual matplotlib colorbar instance
    assert parent.mapping_colorbar is not None
    assert parent.mapping_cax is not None

    # Test setting alpha instantly triggers map redraw
    with patch.object(
        mapping_widget, '_apply_histogram_coloring'
    ) as mock_apply:
        mapping_widget.mesh_alpha_spinbox.setValue(0.73)
        mock_apply.assert_called_once_with("Phase")

    # Disable colorbar
    mapping_widget.mesh_colorbar_checkbox.setChecked(False)
    assert parent.mapping_colorbar is None
    assert parent.mapping_cax is None


def test_mesh_overlay_range_edits_and_sliders(make_viewer_model, qtbot):
    """Text edits to phase/modulation limits should synchronize with the sliders and trigger a redraw."""
    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    mapping_widget = parent.phasor_mapping_tab

    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)
    parent.image_layer_with_phasor_features_combobox.setCurrentText(layer.name)
    parent.on_image_layer_changed()

    mapping_widget.output_mode_combobox.setCurrentText("Phase")
    mapping_widget._on_calculate_lifetime_clicked()
    mapping_widget.mesh_overlay_checkbox.setChecked(True)

    # Inject values simulating manual line edit entries
    mapping_widget.phase_min_edit.setText("0.5")
    mapping_widget.phase_max_edit.setText("1.0")

    with patch.object(
        mapping_widget, '_apply_histogram_coloring'
    ) as mock_apply:
        mapping_widget._on_phase_edits_changed()

        # Verify the slider caught the change and snapped to integers
        min_v, max_v = mapping_widget.phase_range_slider.value()
        assert min_v == int(0.5 * mapping_widget.phase_range_factor)
        assert max_v == int(1.0 * mapping_widget.phase_range_factor)
        mock_apply.assert_called()

    mapping_widget.modulation_min_edit.setText("0.2")
    mapping_widget.modulation_max_edit.setText("0.6")

    with patch.object(
        mapping_widget, '_apply_histogram_coloring'
    ) as mock_apply:
        mapping_widget._on_modulation_edits_changed()

        # Verify the slider caught the change for modulation
        min_v, max_v = mapping_widget.modulation_range_slider.value()
        assert min_v == int(0.2 * mapping_widget.modulation_range_factor)
        assert max_v == int(0.6 * mapping_widget.modulation_range_factor)
        mock_apply.assert_called()


def test_phasor_mapping_teardown_clears_state_when_no_layer(
    make_viewer_model,
    qtbot,
):
    """_teardown_on_layer_change resets all per-layer state when no layer."""
    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    mapping_widget = parent.phasor_mapping_tab

    # Pre-seed some state to ensure teardown clears it
    mapping_widget.lifetime_data = np.array([1.0, 2.0])
    mapping_widget.lifetime_data_original = np.array([1.0, 2.0])
    mapping_widget.current_metric_data = np.array([3.0])
    mapping_widget.current_metric_data_original = np.array([3.0])
    mapping_widget.per_layer_lifetime_data = {"foo": np.array([1.0])}
    mapping_widget.per_layer_lifetime_data_original = {"foo": np.array([1.0])}
    mapping_widget.per_layer_metric_data = {"foo": np.array([1.0])}
    mapping_widget.per_layer_metric_data_original = {"foo": np.array([1.0])}
    mapping_widget.lifetime_layer = object()  # sentinel
    mapping_widget.lifetime_layers = [object()]
    mapping_widget.metric_layers = [object()]

    # No layer added — get_primary_layer_name returns ""
    mapping_widget._teardown_on_layer_change()

    assert mapping_widget.lifetime_data is None
    assert mapping_widget.lifetime_data_original is None
    assert mapping_widget.current_metric_data is None
    assert mapping_widget.current_metric_data_original is None
    assert mapping_widget.per_layer_lifetime_data == {}
    assert mapping_widget.per_layer_lifetime_data_original == {}
    assert mapping_widget.per_layer_metric_data == {}
    assert mapping_widget.per_layer_metric_data_original == {}
    assert mapping_widget.lifetime_layer is None
    assert mapping_widget.lifetime_layers == []
    assert mapping_widget.metric_layers == []


def test_phasor_mapping_teardown_no_op_when_layer_present(
    make_viewer_model, qtbot
):
    """_teardown_on_layer_change does NOT clear state when a layer is present."""
    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)

    mapping_widget = parent.phasor_mapping_tab

    sentinel = np.array([1.0, 2.0, 3.0])
    mapping_widget.lifetime_data = sentinel

    mapping_widget._teardown_on_layer_change()

    # Should NOT have been cleared (layer is present)
    assert mapping_widget.lifetime_data is sentinel


def test_phasor_mapping_restore_with_layer_calls_lifetime_restore(
    make_viewer_model,
    qtbot,
):
    """_restore_on_layer_change runs the with-layer branch."""
    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)

    mapping_widget = parent.phasor_mapping_tab
    mapping_widget._needs_update = True

    with patch.object(
        mapping_widget,
        '_restore_lifetime_settings_from_metadata',
    ) as mock_restore:
        mapping_widget._restore_on_layer_change()
        mock_restore.assert_called_once()

    assert mapping_widget._needs_update is False


def test_phasor_mapping_restore_without_layer_only_clears_flag(
    make_viewer_model,
    qtbot,
):
    """_restore_on_layer_change clears _needs_update even with no layer."""
    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)

    mapping_widget = parent.phasor_mapping_tab
    mapping_widget._needs_update = True

    with patch.object(
        mapping_widget,
        '_restore_lifetime_settings_from_metadata',
    ) as mock_restore:
        mapping_widget._restore_on_layer_change()
        mock_restore.assert_not_called()

    # The flag is still cleared regardless
    assert mapping_widget._needs_update is False


def test_phasor_mapping_apply_histogram_coloring(make_viewer_model, qtbot):
    """Exercise 2D-colormap and mesh-overlay histogram coloring branches."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)
    parent = PlotterWidget(viewer)
    pm = parent.phasor_mapping_tab
    parent.tab_widget.setCurrentWidget(pm)
    parent.plot()

    pm.apply_2d_colormap_checkbox.setChecked(True)
    pm.mesh_overlay_checkbox.setChecked(True)
    pm._apply_histogram_coloring("Phase")
    assert getattr(pm, "_mesh_overlay_imshow", None) is not None
    pm._apply_histogram_coloring("Modulation")
    assert getattr(pm, "_mesh_overlay_imshow", None) is not None

    # Unchecked variants exercise the overlay-removal branches.
    pm.apply_2d_colormap_checkbox.setChecked(False)
    pm.mesh_overlay_checkbox.setChecked(False)
    pm._apply_histogram_coloring("Phase")
    assert getattr(pm, "_mesh_overlay_imshow", None) is None
    assert getattr(pm, "_overlay_imshow", None) is None

    # An invalid output type returns early.
    pm._apply_histogram_coloring("Nope")


def test_phasor_mapping_custom_color_mesh_and_frequency(
    make_viewer_model, qtbot, monkeypatch
):
    """Cover custom-color dialog, mesh-overlay toggle and frequency change."""
    from qtpy.QtGui import QColor
    from qtpy.QtWidgets import QColorDialog

    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    layer.metadata["settings"] = {"frequency": 80.0}
    viewer.add_layer(layer)
    parent = PlotterWidget(viewer)
    pm = parent.phasor_mapping_tab
    parent.tab_widget.setCurrentWidget(pm)
    parent.plot()

    # Custom-color dialog returning a valid colour.
    pm.custom_color_button.setStyleSheet("background-color: rgb(10, 20, 30);")
    monkeypatch.setattr(
        QColorDialog,
        "getColor",
        staticmethod(lambda *a, **k: QColor(200, 100, 50)),
    )
    pm._on_custom_color_clicked()
    assert pm._custom_color.getRgb()[:3] == (200, 100, 50)

    # Mesh overlay toggle in a Phase output mode.
    pm.output_mode_combobox.setCurrentText("Phase")
    pm._on_mesh_overlay_toggled(True)
    pm._on_mesh_overlay_toggled(False)

    # Frequency change in Lifetime mode runs the full recompute path.
    pm.output_mode_combobox.setCurrentText("Lifetime")
    pm.frequency_input.setText("80")
    pm._on_frequency_changed()


def test_phasor_mapping_harmonics_none_fallback(make_viewer_model, qtbot):
    """Cover phasor mapping calculations when harmonics metadata is None (e.g. loaded .R64 files)."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    layer.metadata["settings"] = {"frequency": 80.0}
    # Simulate a .R64 file where harmonics is None and G/S are 2D arrays
    layer.metadata["harmonics"] = None
    layer.metadata["G"] = layer.metadata["G"][0]
    layer.metadata["S"] = layer.metadata["S"][0]
    viewer.add_layer(layer)

    parent = PlotterWidget(viewer)
    pm = parent.phasor_mapping_tab
    parent.tab_widget.setCurrentWidget(pm)
    parent.image_layer_with_phasor_features_combobox.setCurrentText(layer.name)

    # 1. Phase output mode
    pm.output_mode_combobox.setCurrentText("Phase")
    pm.calculate_output_data()
    # Confirm output layers/data created successfully
    assert pm.current_metric_data_original is not None

    # 2. Lifetime output mode
    pm.output_mode_combobox.setCurrentText("Lifetime")
    pm.calculate_output_data()
    # Confirm output layers/data created successfully
    assert pm.current_metric_data_original is not None


def test_phasor_mapping_exceptions(make_viewer_model, qtbot):
    """Test Phasor Mapping module handles missing metadata and IndexError gracefully."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)

    parent = PlotterWidget(viewer)
    pm = parent.phasor_mapping_tab
    parent.tab_widget.setCurrentWidget(pm)
    parent.image_layer_with_phasor_features_combobox.setCurrentText(layer.name)

    # Missing G/S arrays
    layer.metadata["G"] = None
    pm.calculate_output_data()  # Should return early

    # Harmonic not found
    layer.metadata["G"] = np.ones((2, 10, 10))
    layer.metadata["S"] = np.ones((2, 10, 10))
    layer.metadata["harmonics"] = np.array([999])
    pm.calculate_output_data()  # Should return early


def test_resolve_mesh_blur_sigma_zero_display_px_fallback():
    """_resolve_mesh_blur_sigma falls back to the default-resolution ratio
    when the axes report a zero-sized (or unavailable) window extent."""
    ax = MagicMock()
    bbox = MagicMock()
    bbox.width = 0.0
    bbox.height = 0.0
    ax.get_window_extent.return_value = bbox

    resolution = 640
    result = _resolve_mesh_blur_sigma(ax, resolution)

    expected = max(1.5, 1.2 * resolution / _DEFAULT_MESH_RESOLUTION)
    assert result == expected


def test_draw_phasor_mesh_falls_back_when_interpolation_stage_unsupported(
    make_viewer_model, qtbot
):
    """draw_phasor_mesh retries without ``interpolation_stage`` when the
    installed Matplotlib's ``imshow`` doesn't accept that kwarg (older
    Matplotlib versions)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    try:
        real_image = ax.imshow(np.zeros((2, 2)))
        with patch.object(
            ax,
            "imshow",
            side_effect=[TypeError("no interpolation_stage"), real_image],
        ) as mock_imshow:
            result = draw_phasor_mesh(ax, "Phase", resolution=4)

        assert result is real_image
        assert mock_imshow.call_count == 2
    finally:
        plt.close(fig)


def test_get_output_colormap_name_branches():
    """_get_output_colormap_name returns the colormap per output type and
    falls back to 'plasma' for lifetime-style (or any other) outputs."""
    assert PhasorMappingWidget._get_output_colormap_name("Phase") == "cool"
    assert (
        PhasorMappingWidget._get_output_colormap_name("Modulation") == "PiYG"
    )
    assert (
        PhasorMappingWidget._get_output_colormap_name("Normal Lifetime")
        == "plasma"
    )


def test_phasor_mapping_teardown_disconnects_real_layer_events(
    make_viewer_model, qtbot
):
    """_teardown_on_layer_change disconnects colormap/contrast_limits/gamma
    events from real metric layers still present in the viewer when there is
    no primary layer selected."""
    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    mapping_widget = parent.phasor_mapping_tab

    output_layer = Image(
        np.random.rand(10, 10), name="Apparent Phase Lifetime: test"
    )
    viewer.add_layer(output_layer)
    output_layer.events.colormap.connect(mapping_widget._on_colormap_changed)
    output_layer.events.contrast_limits.connect(
        mapping_widget._on_colormap_changed
    )
    output_layer.events.gamma.connect(mapping_widget._on_colormap_changed)
    mapping_widget.metric_layers = [output_layer]

    # No primary layer selected -> get_primary_layer_name() returns "".
    mapping_widget._teardown_on_layer_change()

    assert mapping_widget.metric_layers == []


def test_get_mesh_grid_resolution_exception_fallback(make_viewer_model, qtbot):
    """_get_mesh_grid_resolution falls back to 1000 when
    ``ax.get_window_extent`` raises."""
    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    mapping_widget = parent.phasor_mapping_tab

    ax = MagicMock()
    ax.get_window_extent.side_effect = RuntimeError("boom")

    assert mapping_widget._get_mesh_grid_resolution(ax) == 1000


def test_histogram_datasets_named_after_output_layers(
    make_viewer_model, qtbot
):
    """Histogram datasets are keyed by the analysis output layer name
    (e.g. 'Apparent Phase Lifetime: <image>'), not the intensity image."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)
    parent = PlotterWidget(viewer)
    mt = parent.phasor_mapping_tab
    parent.image_layer_with_phasor_features_combobox.setCurrentText(layer.name)
    mt._on_image_layer_changed()
    mt.frequency_input.setText("80.0")
    mt._on_frequency_changed()
    mt.lifetime_type_combobox.setCurrentText("Apparent Phase Lifetime")
    mt._on_calculate_lifetime_clicked()

    keys = list(mt.histogram_widget._datasets.keys())
    layer_names = {lyr.name for lyr in viewer.layers}
    assert keys, "histogram should have a dataset"
    assert keys[0] != layer.name
    assert keys[0] in layer_names, (keys, layer_names)
    parent.deleteLater()


def test_histogram_multi_layer_datasets_named_after_output_layers(
    make_viewer_model, qtbot
):
    """With multiple selected layers, each output layer feeds its own
    histogram dataset via update_multi_data."""
    viewer = make_viewer_model()
    layer_1 = create_image_layer_with_phasors()
    layer_1.name = "img_one"
    layer_2 = create_image_layer_with_phasors()
    layer_2.name = "img_two"
    viewer.add_layer(layer_1)
    viewer.add_layer(layer_2)
    parent = PlotterWidget(viewer)
    mt = parent.phasor_mapping_tab
    with patch.object(
        parent, "get_selected_layers", return_value=[layer_1, layer_2]
    ):
        mt.frequency_input.setText("80.0")
        mt._on_frequency_changed()
        mt.lifetime_type_combobox.setCurrentText("Apparent Phase Lifetime")
        mt._on_calculate_lifetime_clicked()

    keys = set(mt.histogram_widget._datasets.keys())
    assert keys == {
        "Apparent Phase Lifetime: img_one",
        "Apparent Phase Lifetime: img_two",
    }, keys
    parent.deleteLater()


def test_coloring_box_visibility_follows_output_mode(make_viewer_model, qtbot):
    """The Coloring section is hidden for Lifetime output and shown for
    Phase / Modulation."""
    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    mt = parent.phasor_mapping_tab

    mt.output_mode_combobox.setCurrentText("Lifetime")
    assert mt.coloring_box.isVisible() is False or mt.coloring_box.isHidden()

    mt.output_mode_combobox.setCurrentText("Phase")
    assert not mt.coloring_box.isHidden()

    mt.output_mode_combobox.setCurrentText("Modulation")
    assert not mt.coloring_box.isHidden()

    mt.output_mode_combobox.setCurrentText("Lifetime")
    assert mt.coloring_box.isHidden()
    parent.deleteLater()


def _setup_mapping_selection_workflow(make_napari_viewer, qtbot):
    """Create two selected source layers and calculate a Mapping output."""
    viewer = make_napari_viewer()
    layers = []
    for name in ("mapping_a", "mapping_b"):
        layer = create_image_layer_with_phasors()
        layer.name = name
        viewer.add_layer(layer)
        layers.append(layer)

    parent = PlotterWidget(viewer)
    qtbot.addWidget(parent)
    parent.show()
    parent.image_layers_checkable_combobox.setCheckedItems(
        [layer.name for layer in layers]
    )
    mapping = parent.phasor_mapping_tab
    parent.tab_widget.setCurrentWidget(mapping)
    mapping.frequency_input.setText("80.0")
    mapping._on_calculate_lifetime_clicked()
    mapping.histogram_widget.display_mode = "Individual layers"
    return viewer, parent, mapping, layers


def _click_mapping_source(qtbot, parent, source_name):
    """Toggle a Phasor Layers row through the visible popup."""
    combo = parent.image_layers_checkable_combobox
    row = next(
        row
        for row in range(combo._header_count, combo.model().rowCount())
        if combo.model().item(row).text() == source_name
    )
    combo.showPopup()
    view = combo.view()
    rect = view.visualRect(combo.model().index(row, 0))
    point = rect.center()
    point.setX(rect.left() + 5)
    qtbot.mouseClick(view.viewport(), Qt.LeftButton, pos=point)


def test_mapping_histogram_follows_real_source_selection(
    make_napari_viewer, qtbot
):
    """Mapping curves, statistics, and outputs follow real popup clicks."""
    viewer, parent, mapping, _ = _setup_mapping_selection_workflow(
        make_napari_viewer, qtbot
    )
    output_a = "Apparent Phase Lifetime: mapping_a"
    output_b = "Apparent Phase Lifetime: mapping_b"
    stats = parent.phasor_map_statistics_dock_widget.layer_stats_table

    assert list(mapping.histogram_widget._datasets) == [output_a, output_b]
    assert len(mapping.histogram_widget.ax.lines) == 2
    assert stats.rowCount() == 2

    _click_mapping_source(qtbot, parent, "mapping_b")
    qtbot.waitUntil(
        lambda: parent.get_selected_layer_names() == ["mapping_a"]
        and list(mapping.histogram_widget._datasets) == [output_a],
        timeout=5000,
    )

    assert len(mapping.histogram_widget.ax.lines) == 1
    assert stats.rowCount() == 1
    assert viewer.layers[output_a].visible is True
    assert viewer.layers[output_b].visible is False

    _click_mapping_source(qtbot, parent, "mapping_b")
    qtbot.waitUntil(
        lambda: parent.get_selected_layer_names() == ["mapping_a", "mapping_b"]
        and len(mapping.histogram_widget._datasets) == 2,
        timeout=5000,
    )

    assert viewer.layers[output_b].visible is True
    assert len(mapping.histogram_widget.ax.lines) == 2
    assert stats.rowCount() == 2

    viewer.layers.remove(output_b)
    qtbot.waitUntil(
        lambda: list(mapping.histogram_widget._datasets) == [output_a],
        timeout=5000,
    )

    assert len(mapping.histogram_widget.ax.lines) == 1
    assert stats.rowCount() == 1


def test_mapping_output_controls_refresh_after_first_calculation(
    make_napari_viewer, qtbot
):
    """Parameter and Lifetime Type changes refresh active Mapping output."""
    viewer, _, mapping, _ = _setup_mapping_selection_workflow(
        make_napari_viewer, qtbot
    )

    expected_modes = [
        ("Phase", None, "Phase (rad)"),
        ("Modulation", None, "Modulation"),
        ("Lifetime", "Apparent Modulation Lifetime", "Lifetime (ns)"),
        ("Lifetime", "Normal Lifetime", "Lifetime (ns)"),
    ]
    for mode, lifetime_type, xlabel in expected_modes:
        mapping.output_mode_combobox.setCurrentText(mode)
        if lifetime_type is not None:
            mapping.lifetime_type_combobox.setCurrentText(lifetime_type)
            output_type = lifetime_type
        else:
            output_type = mode

        qtbot.waitUntil(
            lambda output_type=output_type: set(
                mapping.histogram_widget._datasets
            )
            == {
                f"{output_type}: mapping_a",
                f"{output_type}: mapping_b",
            },
            timeout=5000,
        )

        assert mapping.histogram_widget.xlabel == xlabel
        for source_name in ("mapping_a", "mapping_b"):
            output_layer = viewer.layers[f"{output_type}: {source_name}"]
            assert output_layer.metadata['phasor_mapping_output'] == {
                'source_layer': source_name,
                'output_type': output_type,
            }


def test_mapping_output_controls_do_not_calculate_before_first_run(
    make_napari_viewer, qtbot
):
    """Mapping dropdowns stay inert until Calculate succeeds once."""
    viewer = make_napari_viewer()
    layer = create_image_layer_with_phasors()
    layer.name = "mapping_source"
    viewer.add_layer(layer)
    parent = PlotterWidget(viewer)
    mapping = parent.phasor_mapping_tab

    mapping.output_mode_combobox.setCurrentText("Phase")
    mapping.output_mode_combobox.setCurrentText("Lifetime")
    mapping.lifetime_type_combobox.setCurrentText("Normal Lifetime")
    qtbot.wait(200)

    assert mapping._has_calculated_output is False
    assert mapping._mapping_output_layers() == {}
    assert mapping.histogram_widget.counts is None


def test_mapping_and_fret_follow_primary_source_change(
    make_napari_viewer, qtbot
):
    """Primary changes refresh both Mapping and FRET after tab restoration."""
    viewer, parent, mapping, _ = _setup_mapping_selection_workflow(
        make_napari_viewer, qtbot
    )
    fret = parent.fret_tab
    fret.donor_line_edit.setText("2.0")
    fret.frequency_input.setText("80")
    fret.background_real_edit.setText("0.1")
    fret.background_imag_edit.setText("0.1")
    fret.calculate_fret_efficiency_button.click()
    fret.histogram_widget.display_mode = "Individual layers"

    _click_mapping_source(qtbot, parent, "mapping_a")
    qtbot.waitUntil(
        lambda: parent.get_primary_layer_name() == "mapping_b"
        and list(mapping.histogram_widget._datasets)
        == ["Apparent Phase Lifetime: mapping_b"]
        and list(fret.histogram_widget._datasets)
        == ["FRET efficiency: mapping_b"],
        timeout=5000,
    )

    assert viewer.layers["Apparent Phase Lifetime: mapping_a"].visible is False
    assert viewer.layers["FRET efficiency: mapping_a"].visible is False
    assert (
        parent.phasor_map_statistics_dock_widget.layer_stats_table.rowCount()
        == 1
    )
    assert parent.fret_statistics_dock_widget.layer_stats_table.rowCount() == 1


def test_mapping_range_only_changes_selected_outputs(
    make_napari_viewer, qtbot
):
    """Mapping range clipping leaves deselected output data untouched."""
    viewer, parent, mapping, layers = _setup_mapping_selection_workflow(
        make_napari_viewer, qtbot
    )
    output_a = viewer.layers["Apparent Phase Lifetime: mapping_a"]
    output_b = viewer.layers["Apparent Phase Lifetime: mapping_b"]
    output_b_before = output_b.data.copy()
    original_a = layers[0].metadata['derived_data']["Apparent Phase Lifetime"][
        parent.harmonic
    ]

    parent.image_layers_checkable_combobox.setCheckedItems(["mapping_a"])
    parent._layer_selection_timer.stop()
    parent._process_layer_selection_change()
    slider_min, slider_max = mapping.lifetime_range_slider.value()
    quarter = max(1, (slider_max - slider_min) // 4)
    selected_range = (slider_min + quarter, slider_max - quarter)
    mapping._on_lifetime_range_changed(selected_range)

    expected = np.clip(
        original_a,
        selected_range[0] / mapping.lifetime_range_factor,
        selected_range[1] / mapping.lifetime_range_factor,
    )
    np.testing.assert_allclose(output_a.data, expected, equal_nan=True)
    np.testing.assert_array_equal(output_b.data, output_b_before)


def test_mapping_empty_source_selection_clears_histogram(
    make_napari_viewer, qtbot
):
    """Clearing Phasor Layers removes stale Mapping statistics and curves."""
    viewer, parent, mapping, _ = _setup_mapping_selection_workflow(
        make_napari_viewer, qtbot
    )
    parent.image_layers_checkable_combobox.setCheckedItems([])
    parent._layer_selection_timer.stop()
    parent._process_layer_selection_change()

    assert mapping.histogram_widget.counts is None
    assert mapping.histogram_widget._datasets == {}
    assert (
        parent.phasor_map_statistics_dock_widget.layer_stats_table.rowCount()
        == 0
    )
    assert viewer.layers["Apparent Phase Lifetime: mapping_a"].visible is False
    assert viewer.layers["Apparent Phase Lifetime: mapping_b"].visible is False


def test_mapping_invalid_reactive_choice_clears_stale_histogram(
    make_napari_viewer, qtbot
):
    """An active Mapping switch with missing input cannot show old data."""
    _, parent, mapping, _ = _setup_mapping_selection_workflow(
        make_napari_viewer, qtbot
    )
    mapping.output_mode_combobox.setCurrentText("Phase")
    qtbot.waitUntil(
        lambda: bool(mapping.histogram_widget._datasets)
        and all(
            name.startswith("Phase:")
            for name in mapping.histogram_widget._datasets
        ),
        timeout=5000,
    )

    mapping.frequency_input.setText("-")
    mapping.output_mode_combobox.setCurrentText("Lifetime")
    qtbot.waitUntil(
        lambda: mapping.histogram_widget.counts is None,
        timeout=5000,
    )

    assert mapping.histogram_widget._datasets == {}
    assert (
        parent.phasor_map_statistics_dock_widget.layer_stats_table.rowCount()
        == 0
    )


def test_mapping_custom_output_name_survives_rerun_range_and_source_rename(
    make_napari_viewer, qtbot
):
    """Tagged Mapping outputs remain authoritative after manual renaming."""
    viewer, parent, mapping, layers = _setup_mapping_selection_workflow(
        make_napari_viewer, qtbot
    )
    output = viewer.layers["Apparent Phase Lifetime: mapping_a"]
    output.name = "Custom lifetime result"
    output_id = id(output)

    mapping._on_calculate_lifetime_clicked()

    assert id(viewer.layers["Custom lifetime result"]) == output_id
    assert "Apparent Phase Lifetime: mapping_a" not in viewer.layers

    parent.image_layers_checkable_combobox.setCheckedItems(["mapping_a"])
    parent._layer_selection_timer.stop()
    parent._process_layer_selection_change()
    slider_min, slider_max = mapping.lifetime_range_slider.value()
    selected_range = (
        slider_min + max(1, (slider_max - slider_min) // 4),
        slider_max,
    )
    mapping._on_lifetime_range_changed(selected_range)
    original = layers[0].metadata['derived_data']["Apparent Phase Lifetime"][
        parent.harmonic
    ]
    np.testing.assert_allclose(
        output.data,
        np.clip(
            original,
            selected_range[0] / mapping.lifetime_range_factor,
            selected_range[1] / mapping.lifetime_range_factor,
        ),
        equal_nan=True,
    )

    mapping.rename_layer("mapping_a", "mapping_a_renamed")

    assert output.name == "Custom lifetime result"
    assert output.metadata['phasor_mapping_output'] == {
        'source_layer': 'mapping_a_renamed',
        'output_type': 'Apparent Phase Lifetime',
    }


def test_mapping_reactive_coloring_does_not_mutate_previous_metric(
    make_napari_viewer, qtbot
):
    """Phase to Modulation refresh leaves existing Phase layer colors intact."""
    viewer, _, mapping, _ = _setup_mapping_selection_workflow(
        make_napari_viewer, qtbot
    )
    mapping.output_mode_combobox.setCurrentText("Phase")
    qtbot.waitUntil(
        lambda: "Phase: mapping_a" in viewer.layers,
        timeout=5000,
    )
    mapping.apply_2d_colormap_checkbox.setChecked(True)
    phase_layer = viewer.layers["Phase: mapping_a"]
    phase_colors = np.asarray(phase_layer.colormap.colors).copy()

    mapping.output_mode_combobox.setCurrentText("Modulation")
    qtbot.waitUntil(
        lambda: "Modulation: mapping_a" in viewer.layers
        and all(
            mapping._mapping_output_info(layer)[0] == "Modulation"
            for layer in mapping.metric_layers
        ),
        timeout=5000,
    )

    np.testing.assert_array_equal(
        np.asarray(phase_layer.colormap.colors), phase_colors
    )


def test_mapping_frequency_edit_rejects_nonpositive_and_invalid_values(
    make_napari_viewer, qtbot
):
    """Frequency editing uses the same finite-positive validation as Calculate."""
    _, parent, mapping, layers = _setup_mapping_selection_workflow(
        make_napari_viewer, qtbot
    )
    for invalid_value in ("0", "-1", "-"):
        mapping.frequency_input.setText(invalid_value)
        mapping._on_frequency_changed()

        assert mapping.frequency is None
        assert mapping.histogram_widget.counts is None
        assert mapping.histogram_widget._frame_source_datasets == {}

        mapping.frequency_input.setText("80")
        mapping._on_frequency_changed()
        assert mapping.histogram_widget.counts is not None

    parent._broadcast_frequency_value_across_tabs("-1")
    parent._broadcast_frequency_value_across_tabs("not-a-number")

    assert mapping.frequency_input.text() == "80"
    assert layers[0].metadata['settings']['frequency'] == 80.0


def test_mapping_output_mode_syncs_custom_color_button(
    make_viewer_model, qtbot
):
    """Switching outputs keeps the custom-color control consistent."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)
    parent = PlotterWidget(viewer)
    mapping = parent.phasor_mapping_tab

    mapping.output_mode_combobox.setCurrentText("Phase")
    mapping.colormap_combobox.setCurrentText("Select color...")
    assert not mapping.custom_color_button.isHidden()

    mapping.output_mode_combobox.setCurrentText("Modulation")
    assert mapping.colormap_combobox.currentText() == "PiYG"
    assert mapping.custom_color_button.isHidden()

    mapping.colormap_combobox.setCurrentText("Select color...")
    mapping.output_mode_combobox.setCurrentText("Phase")
    assert mapping.colormap_combobox.currentText() == "Select color..."
    assert not mapping.custom_color_button.isHidden()


def test_mapping_defensive_selection_and_legacy_output_helpers(
    make_viewer_model, qtbot
):
    """Defensive selection paths and legacy canonical outputs stay supported."""
    viewer = make_viewer_model()
    source = create_image_layer_with_phasors()
    source.name = "legacy_source"
    viewer.add_layer(source)
    parent = PlotterWidget(viewer)
    mapping = parent.phasor_mapping_tab
    legacy = viewer.add_image(
        np.ones((2, 2)),
        name="Phase: legacy_source",
    )

    assert mapping._mapping_output_info(legacy) == (
        "Phase",
        "legacy_source",
    )

    with patch.object(mapping, "parent_widget", None):
        assert mapping._get_selected_source_names() == set()
    with patch.object(parent, "get_selected_layers", side_effect=RuntimeError):
        assert mapping._get_selected_source_names() == set()
    with patch.object(parent, "get_selected_layers", return_value=[]):
        mapping._sync_mapping_output_visibility()
    assert legacy.visible is False

    mapping.rename_layer("legacy_source", "renamed_source")

    assert legacy.name == "Phase: renamed_source"
    assert legacy.metadata['phasor_mapping_output'] == {
        'source_layer': 'renamed_source',
        'output_type': 'Phase',
    }


def test_mapping_inactive_refresh_and_nonfrequency_edit_are_noops(
    make_viewer_model, qtbot
):
    """Inactive refresh and Phase frequency editing return without calculation."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)
    parent = PlotterWidget(viewer)
    mapping = parent.phasor_mapping_tab

    with patch.object(mapping, "_calculate_and_display_output") as calculate:
        mapping._refresh_active_output()
        mapping.output_mode_combobox.setCurrentText("Phase")
        mapping._on_frequency_changed()
    calculate.assert_not_called()


def test_mapping_calculation_clears_when_selected_layer_has_no_phasors(
    make_viewer_model, qtbot
):
    """A selected layer without phasor arrays cannot reuse previous output."""
    viewer = make_viewer_model()
    valid = create_image_layer_with_phasors()
    viewer.add_layer(valid)
    parent = PlotterWidget(viewer)
    mapping = parent.phasor_mapping_tab
    missing = Image(np.ones((2, 2)), name="missing_phasors")
    mapping.frequency_input.setText("80")

    with (
        patch.object(parent, "get_selected_layers", return_value=[missing]),
        patch.object(parent, "has_phasor_data", return_value=True),
    ):
        assert (
            mapping._calculate_and_display_output(show_warnings=False) is False
        )

    assert mapping.histogram_widget.counts is None


def test_restore_on_layer_change_refreshes_primary_button(
    make_viewer_model, qtbot
):
    """The deferred restore path re-evaluates the primary button so it shows
    the green ready style when the frequency was restored from metadata."""
    from napari_phasors._utils import (
        _PRIMARY_BUTTON_BLOCKED_QSS,
        _PRIMARY_BUTTON_READY_QSS,
    )

    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    layer.metadata["settings"] = {"frequency": 80.0}
    viewer.add_layer(layer)
    parent = PlotterWidget(viewer)
    mt = parent.phasor_mapping_tab

    # Force a stale blocked style, then run the deferred restore path.
    mt.calculate_lifetime_button.setStyleSheet(_PRIMARY_BUTTON_BLOCKED_QSS)
    parent.image_layer_with_phasor_features_combobox.setCurrentText(layer.name)
    mt._restore_on_layer_change()
    assert mt._mapping_validation() is None
    assert (
        mt.calculate_lifetime_button.styleSheet() == _PRIMARY_BUTTON_READY_QSS
    )
    parent.deleteLater()


def _ready_mapping_widget(viewer, name="map_layer"):
    """Return a phasor mapping tab ready to calculate a lifetime map."""
    parent = PlotterWidget(viewer)
    widget = parent.phasor_mapping_tab
    layer = create_image_layer_with_phasors()
    layer.name = name
    viewer.add_layer(layer)
    parent.image_layer_with_phasor_features_combobox.setCurrentText(layer.name)
    widget._on_image_layer_changed()
    widget.frequency_input.setText("80.0")
    widget._on_frequency_changed()
    widget.lifetime_type_combobox.setCurrentText("Apparent Phase Lifetime")
    return parent, widget, layer


def test_mapping_skips_a_layer_without_phasor_arrays(make_viewer_model, qtbot):
    """A layer whose G/S went missing produces no output layer."""
    viewer = make_viewer_model()
    _, widget, layer = _ready_mapping_widget(viewer)
    layer.metadata["S"] = None

    before = len(viewer.layers)
    widget.calculate_output_data()

    assert len(viewer.layers) == before


def test_mapping_skips_a_layer_missing_the_harmonic(make_viewer_model, qtbot):
    """A layer that never computed the selected harmonic is skipped."""
    viewer = make_viewer_model()
    _, widget, layer = _ready_mapping_widget(viewer)
    layer.metadata["harmonics"] = np.array([97])

    before = len(viewer.layers)
    widget.calculate_output_data()

    assert len(viewer.layers) == before


def test_mapping_reports_a_failing_layer(
    make_viewer_model, qtbot, monkeypatch
):
    """A computation that raises is reported by name and output type."""
    viewer = make_viewer_model()
    _, widget, layer = _ready_mapping_widget(viewer)

    errors = []
    monkeypatch.setattr(
        "napari_phasors.phasor_mapping_tab.show_error", errors.append
    )

    def explode(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "napari_phasors.phasor_mapping_tab.phasor_to_apparent_lifetime",
        explode,
    )

    before = len(viewer.layers)
    widget.calculate_output_data()

    assert any("boom" in message for message in errors)
    assert any("map_layer" in message for message in errors)
    assert len(viewer.layers) == before
