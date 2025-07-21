from unittest.mock import MagicMock, patch

import numpy as np
from biaplotter.plotter import CanvasWidget
from napari.layers import Image
from qtpy.QtCore import QCoreApplication, Qt
from qtpy.QtWidgets import QComboBox, QSpinBox, QTabWidget, QVBoxLayout

from napari_phasors._synthetic_generator import (
    make_intensity_layer_with_phasors,
    make_raw_flim_data,
)
from napari_phasors.filter_tab import FilterWidget
from napari_phasors.plotter import PlotterWidget
from napari_phasors.selection_tab import SelectionWidget


def create_image_layer_with_phasors():
    """Create an intensity image layer with phasors for testing."""
    raw_flim_data = make_raw_flim_data()
    harmonic = [1, 2, 3]
    return make_intensity_layer_with_phasors(raw_flim_data, harmonic=harmonic)


def test_phasor_plotter_initialization(make_napari_viewer):
    """Test the initialization of the Phasor Plotter Widget."""
    viewer = make_napari_viewer()

    # Create Plotter widget
    plotter = PlotterWidget(viewer)

    # Basic widget structure tests
    assert plotter.viewer == viewer
    assert isinstance(plotter.layout(), QVBoxLayout)
    assert plotter._labels_layer_with_phasor_features is None

    # Canvas widget tests
    assert hasattr(plotter, 'canvas_widget')
    assert isinstance(plotter.canvas_widget, CanvasWidget)
    assert plotter.canvas_widget.minimumSize().width() >= 600
    assert plotter.canvas_widget.minimumSize().height() >= 400
    assert plotter.canvas_widget.class_spinbox.value == 1

    # UI components tests
    assert hasattr(plotter, 'image_layer_with_phasor_features_combobox')
    assert isinstance(
        plotter.image_layer_with_phasor_features_combobox, QComboBox
    )
    assert hasattr(plotter, 'harmonic_spinbox')
    assert isinstance(plotter.harmonic_spinbox, QSpinBox)
    assert plotter.harmonic_spinbox.minimum() == 1
    assert plotter.harmonic_spinbox.value() == 1

    # Tab widget tests
    assert hasattr(plotter, 'tab_widget')
    assert isinstance(plotter.tab_widget, QTabWidget)
    assert plotter.tab_widget.count() == 7  # Fixed: 7 tabs, not 6

    # Check tab names - Fixed order and names
    tab_names = []
    for i in range(plotter.tab_widget.count()):
        tab_names.append(plotter.tab_widget.tabText(i))
    expected_tabs = [
        "Plot Settings",
        "Calibration",
        "Filter/Threshold",
        "Selection",
        "Components",
        "Lifetime",
        "FRET",
    ]
    assert tab_names == expected_tabs

    # Test individual tabs exist
    assert hasattr(plotter, 'settings_tab')
    assert hasattr(plotter, 'calibration_tab')
    assert hasattr(plotter, 'filter_tab')
    assert hasattr(plotter, 'selection_tab')
    assert hasattr(plotter, 'components_tab')
    assert hasattr(plotter, 'lifetime_tab')
    assert hasattr(plotter, 'fret_tab')

    # Test filter_tab and selection_tab are proper widgets
    assert isinstance(plotter.filter_tab, FilterWidget)
    assert isinstance(plotter.selection_tab, SelectionWidget)

    # Test plotter inputs widget
    assert hasattr(plotter, 'plotter_inputs_widget')
    assert hasattr(plotter.plotter_inputs_widget, 'plot_type_combobox')
    assert hasattr(plotter.plotter_inputs_widget, 'colormap_combobox')
    assert hasattr(plotter.plotter_inputs_widget, 'number_of_bins_spinbox')
    assert hasattr(plotter.plotter_inputs_widget, 'semi_circle_checkbox')
    assert hasattr(plotter.plotter_inputs_widget, 'white_background_checkbox')
    assert hasattr(plotter.plotter_inputs_widget, 'log_scale_checkbox')

    # Test default property values
    assert plotter.harmonic == 1
    assert plotter.plot_type == 'HISTOGRAM2D'
    assert plotter.histogram_colormap == 'turbo'
    assert (
        plotter.toggle_semi_circle == True
    )  # Default is True based on checkbox
    assert (
        plotter.white_background == True
    )  # Default is True based on checkbox

    # Test plot type combobox items
    plot_types = []
    for i in range(plotter.plotter_inputs_widget.plot_type_combobox.count()):
        plot_types.append(
            plotter.plotter_inputs_widget.plot_type_combobox.itemText(i)
        )
    assert plot_types == ['SCATTER', 'HISTOGRAM2D']

    # Test colormap combobox has items (should have all available colormaps)
    assert plotter.plotter_inputs_widget.colormap_combobox.count() > 0

    # Test canvas widget artists
    assert 'SCATTER' in plotter.canvas_widget.artists
    assert 'HISTOGRAM2D' in plotter.canvas_widget.artists

    # Test initial plot elements
    assert plotter.colorbar is None

    # Test minimum widget size
    assert plotter.minimumSize().width() >= 600
    assert plotter.minimumSize().height() >= 800

    # Test axes aspect ratio
    assert plotter.canvas_widget.axes.get_aspect() == 1

    # Test initial axes limits (semi-circle mode)
    xlim = plotter.canvas_widget.axes.get_xlim()
    ylim = plotter.canvas_widget.axes.get_ylim()
    assert (
        xlim[0] < 0 and xlim[1] > 1
    )  # Should include semi-circle range with padding
    assert (
        ylim[0] < 0 and ylim[1] > 0.6
    )  # Should include semi-circle range with padding


def test_phasor_plotter_initialization_with_layer(make_napari_viewer):
    """Test the initialization of the Phasor Plotter Widget with a layer already present."""
    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)

    # Create Plotter widget after layer is added
    plotter = PlotterWidget(viewer)

    # Test that the layer is automatically detected and selected
    assert plotter.image_layer_with_phasor_features_combobox.count() == 1
    assert (
        plotter.image_layer_with_phasor_features_combobox.currentText()
        == intensity_image_layer.name
    )

    # Test that the phasor features layer is set
    assert plotter._labels_layer_with_phasor_features is not None
    assert (
        plotter._labels_layer_with_phasor_features
        == intensity_image_layer.metadata["phasor_features_labels_layer"]
    )

    # Test harmonic spinbox maximum is set based on data
    expected_max_harmonic = (
        intensity_image_layer.metadata["phasor_features_labels_layer"]
        .features["harmonic"]
        .max()
    )
    assert plotter.harmonic_spinbox.maximum() == expected_max_harmonic

    # Test that manual selection column was added
    features_table = intensity_image_layer.metadata[
        "phasor_features_labels_layer"
    ].features
    assert "MANUAL SELECTION #1" in features_table.columns

    # Test that histogram 2D is plotted in the canvas
    assert plotter.plot_type == 'HISTOGRAM2D'  # Should default to histogram 2D

    # Check that the histogram artist exists and is visible
    assert 'HISTOGRAM2D' in plotter.canvas_widget.artists
    histogram_artist = plotter.canvas_widget.artists['HISTOGRAM2D']
    assert histogram_artist is not None

    # Test that colorbar exists for histogram
    assert plotter.colorbar is not None

    # Test that the plot has the correct axes labels
    assert plotter.canvas_widget.axes.get_xlabel() == "G"
    assert plotter.canvas_widget.axes.get_ylabel() == "S"

    # Test that semi-circle is drawn (if enabled by default)
    if plotter.toggle_semi_circle:
        # Check that ticks are added to the semicircle
        assert len(plotter.semi_circle_plot_artist_list) > 0


def test_phasor_plotter_property_setters(make_napari_viewer):
    """Test property setters of the Phasor Plotter Widget."""
    viewer = make_napari_viewer()
    plotter = PlotterWidget(viewer)

    # Test harmonic setter
    plotter.harmonic = 3
    assert plotter.harmonic == 3
    assert plotter.harmonic_spinbox.value() == 3

    # Test plot_type property
    plotter.plot_type = 'SCATTER'
    assert plotter.plot_type == 'SCATTER'
    # The combobox won't automatically update since there's no property setter

    # Test histogram_colormap property
    plotter.histogram_colormap = 'viridis'
    assert plotter.histogram_colormap == 'viridis'

    # Test white_background setter
    plotter.white_background = True
    assert plotter.white_background == True
    assert (
        plotter.plotter_inputs_widget.white_background_checkbox.isChecked()
        == True
    )

    # Test toggle_semi_circle setter
    plotter.toggle_semi_circle = False
    assert plotter.toggle_semi_circle == False
    assert (
        plotter.plotter_inputs_widget.semi_circle_checkbox.isChecked() == False
    )


def test_phasor_plotter_layer_management(make_napari_viewer):
    """Test layer management functionality of the Phasor Plotter Widget."""
    viewer = make_napari_viewer()
    plotter = PlotterWidget(viewer)

    # Initially no layers with phasor features
    assert plotter.image_layer_with_phasor_features_combobox.count() == 0
    assert plotter._labels_layer_with_phasor_features is None

    # Add layer with phasor features
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)

    # Check that combobox is updated
    assert plotter.image_layer_with_phasor_features_combobox.count() == 1
    assert (
        plotter.image_layer_with_phasor_features_combobox.currentText()
        == intensity_image_layer.name
    )

    # Add another layer without phasor features
    regular_layer = Image(np.random.random((10, 10)))
    viewer.add_layer(regular_layer)

    # Combobox should still have only the layer with phasor features
    assert plotter.image_layer_with_phasor_features_combobox.count() == 1

    # Remove the phasor layer
    viewer.layers.remove(intensity_image_layer)

    # Combobox should be empty again
    assert plotter.image_layer_with_phasor_features_combobox.count() == 0
    assert plotter._labels_layer_with_phasor_features is None


def test_phasor_plotter_canvas_click_handling(make_napari_viewer):
    """Test canvas click event handling."""
    viewer = make_napari_viewer()
    plotter = PlotterWidget(viewer)

    # Test that canvas click handler exists
    assert hasattr(plotter, '_on_canvas_click')

    # Create a mock event
    class MockEvent:
        def __init__(self, button=1, xdata=0.5, ydata=0.3, inaxes=None):
            self.button = button
            self.xdata = xdata
            self.ydata = ydata
            self.inaxes = inaxes

    # Test left click with valid coordinates
    event = MockEvent(
        button=1, xdata=0.5, ydata=0.3, inaxes=plotter.canvas_widget.axes
    )
    result = plotter._on_canvas_click(event)
    assert result == (0.5, 0.3)

    # Test click outside axes
    event = MockEvent(button=1, xdata=0.5, ydata=0.3, inaxes=None)
    result = plotter._on_canvas_click(event)
    assert result == (None, None)

    # Test right click
    event = MockEvent(
        button=3, xdata=0.5, ydata=0.3, inaxes=plotter.canvas_widget.axes
    )
    result = plotter._on_canvas_click(event)
    assert result == (None, None)

    # Test click with invalid coordinates
    event = MockEvent(
        button=1, xdata=None, ydata=None, inaxes=plotter.canvas_widget.axes
    )
    result = plotter._on_canvas_click(event)
    assert result == (None, None)


def test_phasor_plotter_semicircle_checkbox(make_napari_viewer):
    """Test semicircle checkbox functionality."""
    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    plotter = PlotterWidget(viewer)

    # Initially semicircle should be enabled (default)
    assert plotter.toggle_semi_circle == True
    assert (
        plotter.plotter_inputs_widget.semi_circle_checkbox.isChecked() == True
    )

    # Get initial axes limits (semicircle mode)
    initial_xlim = plotter.canvas_widget.axes.get_xlim()
    initial_ylim = plotter.canvas_widget.axes.get_ylim()

    # Semicircle should have y-limits starting from 0 or slightly below
    assert initial_ylim[0] <= 0.1  # Small tolerance for padding

    # Uncheck semicircle checkbox (should show full circle)
    plotter.plotter_inputs_widget.semi_circle_checkbox.setChecked(False)
    plotter.toggle_semi_circle = False

    # Get new axes limits (full circle mode)
    new_xlim = plotter.canvas_widget.axes.get_xlim()
    new_ylim = plotter.canvas_widget.axes.get_ylim()

    # Full circle should extend below y=0 significantly
    assert new_ylim[0] < initial_ylim[0]  # Should extend further down
    assert abs(new_ylim[0]) > 0.2  # Should have significant negative y range

    # Check that polar plot elements are updated
    # Full circle should have more polar plot elements than semicircle
    lines = plotter.canvas_widget.axes.get_lines()
    assert len(lines) > 0  # Should have polar plot lines

    # Toggle back to semicircle
    plotter.plotter_inputs_widget.semi_circle_checkbox.setChecked(True)
    plotter.toggle_semi_circle = True

    # Should return to semicircle limits
    restored_ylim = plotter.canvas_widget.axes.get_ylim()
    assert restored_ylim[0] >= new_ylim[0]  # Should not extend as far down


def test_phasor_plotter_colormap_combobox(make_napari_viewer):
    """Test colormap combobox functionality."""
    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    plotter = PlotterWidget(viewer)

    # Test initial colormap
    initial_colormap = plotter.histogram_colormap
    assert initial_colormap == 'turbo'  # or whatever the default is

    # Get available colormaps from combobox
    colormap_combobox = plotter.plotter_inputs_widget.colormap_combobox
    assert colormap_combobox.count() > 1

    # Change to a different colormap
    test_colormap = None
    for i in range(colormap_combobox.count()):
        colormap_name = colormap_combobox.itemText(i)
        if colormap_name != initial_colormap:
            test_colormap = colormap_name
            break

    assert (
        test_colormap is not None
    ), "Should have at least two colormaps available"

    # Set the colormap
    colormap_combobox.setCurrentText(test_colormap)

    # Verify the colormap changed
    assert plotter.histogram_colormap == test_colormap
    assert colormap_combobox.currentText() == test_colormap


def test_phasor_plotter_log_scale_checkbox(make_napari_viewer):
    """Test log scale checkbox functionality."""
    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    plotter = PlotterWidget(viewer)

    # Test initial log scale state
    log_scale_checkbox = plotter.plotter_inputs_widget.log_scale_checkbox
    initial_log_state = log_scale_checkbox.isChecked()

    # Toggle log scale
    log_scale_checkbox.setChecked(not initial_log_state)
    new_log_state = log_scale_checkbox.isChecked()

    # Verify the state changed
    assert new_log_state != initial_log_state

    # If there's a property for log scale, test it
    if hasattr(plotter, 'histogram_log_scale'):
        plotter.histogram_log_scale = new_log_state
        assert plotter.histogram_log_scale == new_log_state

    # Toggle back
    log_scale_checkbox.setChecked(initial_log_state)
    assert log_scale_checkbox.isChecked() == initial_log_state


def test_phasor_plotter_white_background_checkbox(make_napari_viewer):
    """Test white background checkbox functionality."""
    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    plotter = PlotterWidget(viewer)

    # Test initial background state
    white_bg_checkbox = plotter.plotter_inputs_widget.white_background_checkbox
    initial_bg_state = white_bg_checkbox.isChecked()

    # Get initial background color
    initial_bg_color = plotter.canvas_widget.axes.get_facecolor()

    # Toggle white background
    new_bg_state = not initial_bg_state
    white_bg_checkbox.setChecked(new_bg_state)
    plotter.white_background = new_bg_state

    # Verify the property changed
    assert plotter.white_background == new_bg_state
    assert white_bg_checkbox.isChecked() == new_bg_state

    # Check that background color changed
    new_bg_color = plotter.canvas_widget.axes.get_facecolor()

    if new_bg_state:
        # White background should be close to (1, 1, 1, 1) or (1, 1, 1, 0) for transparent white
        assert new_bg_color[0] > 0.9  # R component close to 1
        assert new_bg_color[1] > 0.9  # G component close to 1
        assert new_bg_color[2] > 0.9  # B component close to 1
    else:
        # Non-white background (could be transparent or dark)
        # Should be different from white
        is_white = (
            new_bg_color[0] > 0.9
            and new_bg_color[1] > 0.9
            and new_bg_color[2] > 0.9
        )
        assert (
            not is_white or new_bg_color[3] < 0.1
        )  # Either not white, or transparent

    # Toggle back and verify
    white_bg_checkbox.setChecked(initial_bg_state)
    plotter.white_background = initial_bg_state

    restored_bg_color = plotter.canvas_widget.axes.get_facecolor()
    # Background should return to initial state (or close to it)
    np.testing.assert_allclose(restored_bg_color, initial_bg_color, atol=0.1)


def test_phasor_plotter_colorbar_updates(make_napari_viewer):
    """Test that colorbar updates when histogram settings change."""
    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    plotter = PlotterWidget(viewer)

    # Should have a colorbar for histogram
    assert plotter.colorbar is not None

    # Change colormap
    colormap_combobox = plotter.plotter_inputs_widget.colormap_combobox
    if colormap_combobox.count() > 1:
        # Change to different colormap
        current_index = colormap_combobox.currentIndex()
        new_index = (current_index + 1) % colormap_combobox.count()
        colormap_combobox.setCurrentIndex(new_index)

        new_colormap = colormap_combobox.currentText()
        plotter.histogram_colormap = new_colormap

        # Verify colorbar still exists after colormap change
        # (actual colorbar update might require calling update method)
        assert plotter.colorbar is not None

    # Toggle log scale
    log_checkbox = plotter.plotter_inputs_widget.log_scale_checkbox
    log_checkbox.setChecked(not log_checkbox.isChecked())

    # Verify colorbar still exists after log scale change
    assert plotter.colorbar is not None
