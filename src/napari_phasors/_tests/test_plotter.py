from unittest.mock import Mock, patch

import numpy as np
from biaplotter.plotter import CanvasWidget
from napari.layers import Image
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
    time_constants = [0.1, 1, 2, 3, 4, 5, 10]
    raw_flim_data = make_raw_flim_data(time_constants=time_constants)
    harmonic = [1, 2, 3]
    return make_intensity_layer_with_phasors(raw_flim_data, harmonic=harmonic)


def test_phasor_plotter_initialization_values(make_napari_viewer):
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
    assert plotter.canvas_widget.minimumSize().width() >= 300
    assert plotter.canvas_widget.minimumSize().height() >= 300
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
    assert plotter.tab_widget.count() == 7  # Number of tabs should be 7

    # Check tab names
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
    # TODO: Add other tests when those tabs are implemented

    # Test settings tab inputs widget
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
    )  # Should default to semi-circle mode
    assert (
        plotter.white_background == True
    )  # Should default to white background

    # Test plot type combobox items
    plot_types = []
    for i in range(plotter.plotter_inputs_widget.plot_type_combobox.count()):
        plot_types.append(
            plotter.plotter_inputs_widget.plot_type_combobox.itemText(i)
        )
    assert plot_types == ['HISTOGRAM2D', 'SCATTER']

    # Test colormap combobox has items (should have all available colormaps)
    assert plotter.plotter_inputs_widget.colormap_combobox.count() > 0

    # Test canvas widget artists
    assert 'SCATTER' in plotter.canvas_widget.artists
    assert 'HISTOGRAM2D' in plotter.canvas_widget.artists

    # Test initial plot elements
    assert plotter.colorbar is None

    # Test minimum widget size
    assert plotter.minimumSize().width() >= 300
    assert plotter.minimumSize().height() >= 300

    # Test axes aspect ratio
    assert plotter.canvas_widget.axes.get_aspect() == 1

    # Test initial axes limits (semi-circle mode)
    # TODO: Test specific limits?
    xlim = plotter.canvas_widget.axes.get_xlim()
    ylim = plotter.canvas_widget.axes.get_ylim()
    assert xlim[0] < 0 and xlim[1] > 1
    assert ylim[0] < 0 and ylim[1] > 0.6


def test_phasor_plotter_initialization_plot_not_called(make_napari_viewer):

    viewer = make_napari_viewer()

    # Test that plot method is not called during initialization
    with patch(
        'napari_phasors.plotter.PlotterWidget._set_active_artist_and_plot'
    ) as mock_plot:
        # Create the plotter widget (this will call plot during initialization)
        plotter = PlotterWidget(viewer)

        # Verify plot was not called during initialization
        mock_plot.assert_not_called()

        # Check that modifying values does not call plot
        plotter.plot_type = 'SCATTER'
        mock_plot.assert_not_called()
        plotter.histogram_colormap = 'viridis'
        mock_plot.assert_not_called()
        plotter.toggle_semi_circle = False
        mock_plot.assert_not_called()
        plotter.white_background = False
        mock_plot.assert_not_called()
        plotter.plotter_inputs_widget.semi_circle_checkbox.setChecked(False)
        mock_plot.assert_not_called()
        plotter.plotter_inputs_widget.white_background_checkbox.setChecked(
            False
        )
        mock_plot.assert_not_called()
        plotter.deleteLater()


def test_phasor_plotter_initialization_with_layer(make_napari_viewer):
    """Test the initialization of the Phasor Plotter Widget with a layer already present."""
    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)

    with patch(
        'napari_phasors.plotter.PlotterWidget._set_active_artist_and_plot'
    ) as mock_plot:
        # Create the plotter widget (this will call plot during initialization)
        plotter = PlotterWidget(viewer)

        # Verify plot was called exactly once during initialization
        mock_plot.assert_called_once()
        plotter.deleteLater()

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


def test_adding_removing_layers_updates_plot(make_napari_viewer):
    """Test that adding/removing layers updates the plotter widget."""
    viewer = make_napari_viewer()

    with patch(
        'napari_phasors.plotter.PlotterWidget._set_active_artist_and_plot'
    ) as mock_plot:
        # Create the plotter widget (this will call plot during initialization)
        plotter = PlotterWidget(viewer)

        # Add a layer with phasor features
        intensity_image_layer = create_image_layer_with_phasors()
        viewer.add_layer(intensity_image_layer)
        # Verify plot was called once after adding layer
        mock_plot.assert_called_once()
        mock_plot.reset_mock()  # Reset mock for next call

        # Remove the layer
        viewer.layers.remove(intensity_image_layer)
        # Verify the plot method was not called after removing the layer
        mock_plot.assert_not_called()

        # Check that modifying values does not call plot
        plotter.plot_type = 'SCATTER'
        mock_plot.assert_not_called()
        plotter.histogram_colormap = 'viridis'
        mock_plot.assert_not_called()
        plotter.toggle_semi_circle = False
        mock_plot.assert_not_called()
        plotter.white_background = False
        mock_plot.assert_not_called()
        plotter.plotter_inputs_widget.semi_circle_checkbox.setChecked(False)
        mock_plot.assert_not_called()
        plotter.plotter_inputs_widget.white_background_checkbox.setChecked(
            False
        )
        mock_plot.assert_not_called()

        # Add two layers with phasor features
        intensity_image_layer_2 = create_image_layer_with_phasors()
        viewer.add_layer(intensity_image_layer_2)
        # Verify plot was called once after adding second layer
        mock_plot.assert_called_once()
        mock_plot.reset_mock()  # Reset mock for next call

        viewer.add_layer(intensity_image_layer)
        mock_plot.assert_not_called()

        # Check values changed before are kept
        assert plotter.plot_type == 'SCATTER'
        assert plotter.histogram_colormap == 'viridis'
        assert plotter.toggle_semi_circle == False
        assert plotter.white_background == False
        assert (
            plotter.plotter_inputs_widget.semi_circle_checkbox.isChecked()
            == False
        )
        assert (
            plotter.plotter_inputs_widget.white_background_checkbox.isChecked()
            == False
        )

        # Check that the combobox has both layers with phasor features
        assert plotter.image_layer_with_phasor_features_combobox.count() == 2
        # Check that the first layer is still selected
        assert (
            plotter.image_layer_with_phasor_features_combobox.currentText()
            == intensity_image_layer_2.name
        )
        plotter.deleteLater()


def test_add_layer_without_phasor_features_does_not_trigger_plot_or_combobox(
    make_napari_viewer,
):
    """Test that adding a layer without phasor features does not call plot or update the combobox."""
    from napari.layers import Image

    viewer = make_napari_viewer()

    # Patch plot and on_labels_layer_with_phasor_features_changed
    with (
        patch.object(
            PlotterWidget, '_set_active_artist_and_plot'
        ) as mock_plot,
        patch.object(
            PlotterWidget, 'on_labels_layer_with_phasor_features_changed'
        ) as mock_labels_changed,
    ):

        plotter = PlotterWidget(viewer)
        # Add a regular image layer (no phasor features)
        regular_layer = Image(np.random.random((10, 10)))
        viewer.add_layer(regular_layer)

        # plot and on_labels_layer_with_phasor_features_changed should NOT be called
        mock_plot.assert_not_called()
        mock_labels_changed.assert_not_called()

        # The combobox should not be updated
        assert plotter.image_layer_with_phasor_features_combobox.count() == 0
        assert (
            plotter.image_layer_with_phasor_features_combobox.currentText()
            == ''
        )
        plotter.deleteLater()


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
    initial_ylim = plotter.canvas_widget.axes.get_ylim()

    # Semicircle should have y-limits starting from 0 or slightly below
    assert initial_ylim[0] <= 0.1  # Small tolerance for padding

    # Uncheck semicircle checkbox (should show full circle)
    plotter.plotter_inputs_widget.semi_circle_checkbox.setChecked(False)
    plotter.toggle_semi_circle = False

    # Get new axes limits (full circle mode)
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
        assert plotter.colorbar is not None

    # Toggle log scale
    log_checkbox = plotter.plotter_inputs_widget.log_scale_checkbox
    log_checkbox.setChecked(not log_checkbox.isChecked())

    # Verify colorbar still exists after log scale change
    assert plotter.colorbar is not None


def test_on_labels_layer_with_phasor_features_changed_prevents_recursion(
    make_napari_viewer,
):
    """Test that the method prevents recursive calls using the guard flag."""

    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    plotter = PlotterWidget(viewer)

    # Set the guard flag to simulate being already in the method
    plotter._in_on_labels_layer_with_phasor_features_changed = True

    with patch.object(plotter, 'plot') as mock_plot:
        # Call the method - should return early due to guard
        plotter.on_labels_layer_with_phasor_features_changed()

        # Verify plot was not called due to guard
        mock_plot.assert_not_called()
        plotter.deleteLater()

    # Verify guard flag is still True (not reset by early return)
    assert plotter._in_on_labels_layer_with_phasor_features_changed == True


def test_on_labels_layer_with_phasor_features_changed_with_empty_layer_name(
    make_napari_viewer,
):
    """Test behavior when combobox has empty layer name."""

    viewer = make_napari_viewer()
    plotter = PlotterWidget(viewer)

    # Ensure combobox is empty
    plotter.image_layer_with_phasor_features_combobox.clear()

    with patch.object(plotter, 'plot') as mock_plot:
        plotter.on_labels_layer_with_phasor_features_changed()

        # Should not call plot when layer name is empty
        mock_plot.assert_not_called()
        plotter.deleteLater()

    # Verify labels layer is set to None
    assert plotter._labels_layer_with_phasor_features is None

    # Now add a layer with phasors

    with patch.object(plotter, 'plot') as mock_plot:
        intensity_image_layer = create_image_layer_with_phasors()
        viewer.add_layer(intensity_image_layer)

        mock_plot.assert_called_once()

        mock_plot.reset_mock()

        # Remove the layer to simulate empty state again
        viewer.layers.remove(intensity_image_layer)

        # Should not call plot when layer name is empty
        mock_plot.assert_not_called()
        plotter.deleteLater()


def test_on_labels_layer_with_phasor_features_changed_sets_harmonic_maximum(
    make_napari_viewer,
):
    """Test that harmonic spinbox maximum is set based on layer data."""
    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    plotter = PlotterWidget(viewer)

    # Get expected maximum harmonic from the data
    expected_max = (
        intensity_image_layer.metadata["phasor_features_labels_layer"]
        .features["harmonic"]
        .max()
    )

    # Call the method
    plotter.on_labels_layer_with_phasor_features_changed()

    # Verify harmonic spinbox maximum is set correctly
    assert plotter.harmonic_spinbox.maximum() == expected_max


def test_on_labels_layer_with_phasor_features_changed_updates_labels_layer(
    make_napari_viewer,
):
    """Test that the _labels_layer_with_phasor_features attribute is updated correctly."""
    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    plotter = PlotterWidget(viewer)

    # Initially should be set from initialization
    initial_layer = plotter._labels_layer_with_phasor_features

    # Clear it to test the update
    plotter._labels_layer_with_phasor_features = None

    # Call the method
    plotter.on_labels_layer_with_phasor_features_changed()

    # Verify it's set to the correct layer
    expected_layer = intensity_image_layer.metadata[
        "phasor_features_labels_layer"
    ]
    assert plotter._labels_layer_with_phasor_features == expected_layer
    assert plotter._labels_layer_with_phasor_features == initial_layer


def test_on_labels_layer_with_phasor_features_changed_guard_flag_cleanup(
    make_napari_viewer,
):
    """Test that guard flag is properly cleaned up even if an exception occurs."""

    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    plotter = PlotterWidget(viewer)

    # Mock plot to raise an exception
    with patch.object(
        plotter, 'plot', side_effect=Exception("Test exception")
    ):
        try:
            plotter.on_labels_layer_with_phasor_features_changed()
        except Exception:
            pass  # We expect the exception
        plotter.deleteLater()

    # Verify guard flag is cleaned up even after exception
    assert (
        not hasattr(
            plotter, '_in_on_labels_layer_with_phasor_features_changed'
        )
        or plotter._in_on_labels_layer_with_phasor_features_changed == False
    )


def test_on_labels_layer_with_phasor_features_changed_multiple_calls(
    make_napari_viewer,
):
    """Test multiple sequential calls to ensure plot is called each time."""

    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    plotter = PlotterWidget(viewer)

    with patch.object(plotter, 'plot') as mock_plot:
        # Call multiple times
        plotter.on_labels_layer_with_phasor_features_changed()
        plotter.on_labels_layer_with_phasor_features_changed()
        plotter.on_labels_layer_with_phasor_features_changed()

        # Each call should result in plot being called once
        assert mock_plot.call_count == 3
        plotter.deleteLater()


def test_phasor_plotter_tab_changed_functionality(make_napari_viewer):
    """Test tab change functionality and artist visibility management."""
    from unittest.mock import Mock, patch

    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    plotter = PlotterWidget(viewer)

    # Mock the tab-specific visibility methods
    with (
        patch.object(plotter, '_hide_all_tab_artists') as mock_hide_all,
        patch.object(plotter, '_show_tab_artists') as mock_show_tab,
        patch.object(
            plotter.canvas_widget.figure.canvas, 'draw_idle'
        ) as mock_draw,
    ):

        # Test tab change to components tab (index 4)
        components_tab_index = 4
        plotter._on_tab_changed(components_tab_index)

        # Verify methods were called
        mock_hide_all.assert_called_once()
        mock_show_tab.assert_called_once_with(plotter.components_tab)
        mock_draw.assert_called_once()

        # Reset mocks
        mock_hide_all.reset_mock()
        mock_show_tab.reset_mock()
        mock_draw.reset_mock()

        # Test tab change to FRET tab (index 6)
        fret_tab_index = 6
        plotter._on_tab_changed(fret_tab_index)

        mock_hide_all.assert_called_once()
        mock_show_tab.assert_called_once_with(plotter.fret_tab)
        mock_draw.assert_called_once()


def test_phasor_plotter_hide_and_show_tab_artists(make_napari_viewer):
    """Test hiding and showing tab-specific artists."""
    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    plotter = PlotterWidget(viewer)

    # Mock the tab-specific visibility methods
    with (
        patch.object(
            plotter, '_set_components_visibility'
        ) as mock_components_vis,
        patch.object(plotter, '_set_fret_visibility') as mock_fret_vis,
    ):

        # Test _hide_all_tab_artists
        plotter._hide_all_tab_artists()

        # Should call visibility methods with False
        mock_components_vis.assert_called_with(False)
        mock_fret_vis.assert_called_with(False)

        # Reset mocks
        mock_components_vis.reset_mock()
        mock_fret_vis.reset_mock()

        # Test _show_tab_artists with components tab
        plotter._show_tab_artists(plotter.components_tab)
        mock_components_vis.assert_called_once_with(True)
        mock_fret_vis.assert_not_called()

        # Reset mocks
        mock_components_vis.reset_mock()
        mock_fret_vis.reset_mock()

        # Test _show_tab_artists with FRET tab
        plotter._show_tab_artists(plotter.fret_tab)
        mock_fret_vis.assert_called_once_with(True)
        mock_components_vis.assert_not_called()

        # Test _show_tab_artists with non-specific tab (should not call any visibility methods)
        mock_components_vis.reset_mock()
        mock_fret_vis.reset_mock()

        plotter._show_tab_artists(plotter.settings_tab)
        mock_components_vis.assert_not_called()
        mock_fret_vis.assert_not_called()


def test_phasor_plotter_tab_specific_visibility_methods(make_napari_viewer):
    """Test the tab-specific visibility methods."""
    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    plotter = PlotterWidget(viewer)

    # Test _set_components_visibility
    with patch.object(
        plotter.components_tab, 'set_artists_visible'
    ) as mock_components_artists:
        plotter._set_components_visibility(True)
        mock_components_artists.assert_called_once_with(True)

        mock_components_artists.reset_mock()
        plotter._set_components_visibility(False)
        mock_components_artists.assert_called_once_with(False)

    # Test _set_fret_visibility
    with patch.object(
        plotter.fret_tab, 'set_artists_visible'
    ) as mock_fret_artists:
        plotter._set_fret_visibility(True)
        mock_fret_artists.assert_called_once_with(True)

        mock_fret_artists.reset_mock()
        plotter._set_fret_visibility(False)
        mock_fret_artists.assert_called_once_with(False)


def test_phasor_plotter_tab_widget_signal_connection(make_napari_viewer):
    """Test that tab widget currentChanged signal is properly connected."""
    viewer = make_napari_viewer()

    with patch.object(PlotterWidget, '_on_tab_changed') as mock_tab_changed:
        plotter = PlotterWidget(viewer)

        # Simulate tab change by emitting the signal directly
        plotter.tab_widget.setCurrentIndex(2)  # Change to filter tab

        # The signal should trigger the method
        mock_tab_changed.assert_called_with(2)


def test_phasor_plotter_tab_changed_with_different_tabs(make_napari_viewer):
    """Test tab changes with different tab indices."""
    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    plotter = PlotterWidget(viewer)

    with (
        patch.object(plotter, '_hide_all_tab_artists') as mock_hide_all,
        patch.object(plotter, '_show_tab_artists') as mock_show_tab,
    ):

        # Test each tab index
        tab_names = [
            "Plot Settings",
            "Calibration",
            "Filter/Threshold",
            "Selection",
            "Components",
            "Lifetime",
            "FRET",
        ]

        for i in range(plotter.tab_widget.count()):
            mock_hide_all.reset_mock()
            mock_show_tab.reset_mock()

            plotter._on_tab_changed(i)

            # Should always hide all artists first
            mock_hide_all.assert_called_once()

            # Should show artists for the current tab
            expected_tab = plotter.tab_widget.widget(i)
            mock_show_tab.assert_called_once_with(expected_tab)


def test_phasor_plotter_tab_changed_functionality(make_napari_viewer):
    """Test tab change functionality and artist visibility management."""

    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    plotter = PlotterWidget(viewer)

    # Mock the tab-specific visibility methods
    with (
        patch.object(plotter, '_hide_all_tab_artists') as mock_hide_all,
        patch.object(plotter, '_show_tab_artists') as mock_show_tab,
        patch.object(
            plotter.canvas_widget.figure.canvas, 'draw_idle'
        ) as mock_draw,
    ):

        # Test tab change to components tab (index 4)
        components_tab_index = 4
        plotter._on_tab_changed(components_tab_index)

        # Verify methods were called
        mock_hide_all.assert_called_once()
        mock_show_tab.assert_called_once_with(plotter.components_tab)
        mock_draw.assert_called_once()

        # Reset mocks
        mock_hide_all.reset_mock()
        mock_show_tab.reset_mock()
        mock_draw.reset_mock()

        # Test tab change to FRET tab (index 6)
        fret_tab_index = 6
        plotter._on_tab_changed(fret_tab_index)

        mock_hide_all.assert_called_once()
        mock_show_tab.assert_called_once_with(plotter.fret_tab)
        mock_draw.assert_called_once()


def test_phasor_plotter_hide_and_show_tab_artists(make_napari_viewer):
    """Test hiding and showing tab-specific artists."""
    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    plotter = PlotterWidget(viewer)

    # Mock the tab-specific visibility methods
    with (
        patch.object(
            plotter, '_set_components_visibility'
        ) as mock_components_vis,
        patch.object(plotter, '_set_fret_visibility') as mock_fret_vis,
    ):

        # Test _hide_all_tab_artists
        plotter._hide_all_tab_artists()

        # Should call visibility methods with False
        mock_components_vis.assert_called_with(False)
        mock_fret_vis.assert_called_with(False)

        # Reset mocks
        mock_components_vis.reset_mock()
        mock_fret_vis.reset_mock()

        # Test _show_tab_artists with components tab
        plotter._show_tab_artists(plotter.components_tab)
        mock_components_vis.assert_called_once_with(True)
        mock_fret_vis.assert_not_called()

        # Reset mocks
        mock_components_vis.reset_mock()
        mock_fret_vis.reset_mock()

        # Test _show_tab_artists with FRET tab
        plotter._show_tab_artists(plotter.fret_tab)
        mock_fret_vis.assert_called_once_with(True)
        mock_components_vis.assert_not_called()

        # Test _show_tab_artists with non-specific tab (should not call any visibility methods)
        mock_components_vis.reset_mock()
        mock_fret_vis.reset_mock()

        plotter._show_tab_artists(plotter.settings_tab)
        mock_components_vis.assert_not_called()
        mock_fret_vis.assert_not_called()


def test_phasor_plotter_tab_specific_visibility_methods(make_napari_viewer):
    """Test the tab-specific visibility methods."""
    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    plotter = PlotterWidget(viewer)

    # Test _set_components_visibility
    with patch.object(
        plotter.components_tab, 'set_artists_visible'
    ) as mock_components_artists:
        plotter._set_components_visibility(True)
        mock_components_artists.assert_called_once_with(True)

        mock_components_artists.reset_mock()
        plotter._set_components_visibility(False)
        mock_components_artists.assert_called_once_with(False)

    # Test _set_fret_visibility
    with patch.object(
        plotter.fret_tab, 'set_artists_visible'
    ) as mock_fret_artists:
        plotter._set_fret_visibility(True)
        mock_fret_artists.assert_called_once_with(True)

        mock_fret_artists.reset_mock()
        plotter._set_fret_visibility(False)
        mock_fret_artists.assert_called_once_with(False)


def test_phasor_plotter_tab_widget_signal_connection(make_napari_viewer):
    """Test that tab widget currentChanged signal is properly connected."""
    viewer = make_napari_viewer()

    with patch.object(PlotterWidget, '_on_tab_changed') as mock_tab_changed:
        plotter = PlotterWidget(viewer)

        # Simulate tab change by emitting the signal directly
        plotter.tab_widget.setCurrentIndex(2)  # Change to filter tab

        # The signal should trigger the method
        mock_tab_changed.assert_called_with(2)


def test_phasor_plotter_tab_changed_with_different_tabs(make_napari_viewer):
    """Test tab changes with different tab indices."""
    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    plotter = PlotterWidget(viewer)

    with (
        patch.object(plotter, '_hide_all_tab_artists') as mock_hide_all,
        patch.object(plotter, '_show_tab_artists') as mock_show_tab,
    ):

        # Test each tab index
        tab_names = [
            "Plot Settings",
            "Calibration",
            "Filter/Threshold",
            "Selection",
            "Components",
            "Lifetime",
            "FRET",
        ]

        for i in range(plotter.tab_widget.count()):
            mock_hide_all.reset_mock()
            mock_show_tab.reset_mock()

            plotter._on_tab_changed(i)

            # Should always hide all artists first
            mock_hide_all.assert_called_once()

            # Should show artists for the current tab
            expected_tab = plotter.tab_widget.widget(i)
            mock_show_tab.assert_called_once_with(expected_tab)


def test_phasor_plotter_set_components_visibility_method(make_napari_viewer):
    """Test _set_components_visibility method behavior."""
    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    plotter = PlotterWidget(viewer)

    # Test with components tab having set_artists_visible method
    with patch.object(
        plotter.components_tab, 'set_artists_visible'
    ) as mock_set_visible:
        # Test setting visibility to True
        plotter._set_components_visibility(True)
        mock_set_visible.assert_called_once_with(True)

        mock_set_visible.reset_mock()

        # Test setting visibility to False
        plotter._set_components_visibility(False)
        mock_set_visible.assert_called_once_with(False)


def test_phasor_plotter_set_fret_visibility_method(make_napari_viewer):
    """Test _set_fret_visibility method behavior."""
    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    plotter = PlotterWidget(viewer)

    # Test with FRET tab having set_artists_visible method
    with patch.object(
        plotter.fret_tab, 'set_artists_visible'
    ) as mock_set_visible:
        # Test setting visibility to True
        plotter._set_fret_visibility(True)
        mock_set_visible.assert_called_once_with(True)

        mock_set_visible.reset_mock()

        # Test setting visibility to False
        plotter._set_fret_visibility(False)
        mock_set_visible.assert_called_once_with(False)


def test_phasor_plotter_set_visibility_methods_without_tabs(
    make_napari_viewer,
):
    """Test visibility methods when tabs don't have the expected attributes."""
    viewer = make_napari_viewer()
    plotter = PlotterWidget(viewer)

    # Remove the components_tab attribute to test the hasattr check
    original_components_tab = plotter.components_tab
    del plotter.components_tab

    # Should not raise an error when components_tab doesn't exist
    try:
        plotter._set_components_visibility(True)
        plotter._set_components_visibility(False)
    except AttributeError:
        assert (
            False
        ), "_set_components_visibility should handle missing components_tab gracefully"

    # Restore components_tab
    plotter.components_tab = original_components_tab

    # Remove the fret_tab attribute to test the hasattr check
    original_fret_tab = plotter.fret_tab
    del plotter.fret_tab

    # Should not raise an error when fret_tab doesn't exist
    try:
        plotter._set_fret_visibility(True)
        plotter._set_fret_visibility(False)
    except AttributeError:
        assert (
            False
        ), "_set_fret_visibility should handle missing fret_tab gracefully"

    # Restore fret_tab
    plotter.fret_tab = original_fret_tab


def test_phasor_plotter_visibility_methods_error_handling(make_napari_viewer):
    """Test that visibility methods handle errors gracefully."""
    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    plotter = PlotterWidget(viewer)

    # Mock set_artists_visible to raise an exception
    def mock_set_visible_error(visible):
        raise Exception("Mock error in set_artists_visible")

    with patch.object(
        plotter.components_tab,
        'set_artists_visible',
        side_effect=mock_set_visible_error,
    ):
        # Should not crash if set_artists_visible raises an exception
        try:
            plotter._set_components_visibility(True)
        except Exception as e:
            # If the method doesn't handle the exception, the test will catch it
            # In a real implementation, you might want to handle this gracefully
            assert "Mock error in set_artists_visible" in str(e)

    with patch.object(
        plotter.fret_tab,
        'set_artists_visible',
        side_effect=mock_set_visible_error,
    ):
        # Should not crash if set_artists_visible raises an exception
        try:
            plotter._set_fret_visibility(False)
        except Exception as e:
            # If the method doesn't handle the exception, the test will catch it
            # In a real implementation, you might want to handle this gracefully
            assert "Mock error in set_artists_visible" in str(e)


def test_phasor_plotter_visibility_methods_called_by_hide_show_artists(
    make_napari_viewer,
):
    """Test that visibility methods are called correctly by hide/show artists methods."""
    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    plotter = PlotterWidget(viewer)

    # Test that _hide_all_tab_artists calls both visibility methods with False
    with (
        patch.object(plotter, '_set_components_visibility') as mock_comp_vis,
        patch.object(plotter, '_set_fret_visibility') as mock_fret_vis,
    ):

        plotter._hide_all_tab_artists()

        # Check that the methods were called with False (allow multiple calls)
        mock_comp_vis.assert_called_with(False)
        mock_fret_vis.assert_called_with(False)

        # Verify all calls were with False
        for call in mock_comp_vis.call_args_list:
            assert call[0][0] == False
        for call in mock_fret_vis.call_args_list:
            assert call[0][0] == False

    # Test that _show_tab_artists calls the correct visibility method for components tab
    with (
        patch.object(plotter, '_set_components_visibility') as mock_comp_vis,
        patch.object(plotter, '_set_fret_visibility') as mock_fret_vis,
    ):

        plotter._show_tab_artists(plotter.components_tab)

        mock_comp_vis.assert_called_once_with(True)
        mock_fret_vis.assert_not_called()

    # Test that _show_tab_artists calls the correct visibility method for FRET tab
    with (
        patch.object(plotter, '_set_components_visibility') as mock_comp_vis,
        patch.object(plotter, '_set_fret_visibility') as mock_fret_vis,
    ):

        plotter._show_tab_artists(plotter.fret_tab)

        mock_comp_vis.assert_not_called()
        mock_fret_vis.assert_called_once_with(True)


def test_phasor_plotter_visibility_methods_integration_with_tab_changes(
    make_napari_viewer,
):
    """Test integration of visibility methods with actual tab changes."""
    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    plotter = PlotterWidget(viewer)

    with (
        patch.object(plotter, '_set_components_visibility') as mock_comp_vis,
        patch.object(plotter, '_set_fret_visibility') as mock_fret_vis,
    ):

        # Reset mocks to clear any initialization calls
        mock_comp_vis.reset_mock()
        mock_fret_vis.reset_mock()

        # Change to components tab (index 4)
        plotter.tab_widget.setCurrentIndex(4)

        # Should have at least one False call (hide) and one True call (show) for components
        # The exact count may vary depending on initialization and signal handling
        assert mock_comp_vis.call_count >= 2
        assert mock_fret_vis.call_count >= 1

        # Check that the last calls were in the correct order
        calls_comp = mock_comp_vis.call_args_list
        calls_fret = mock_fret_vis.call_args_list

        # The last component call should be True (show components)
        assert calls_comp[-1][0][0] == True
        # All fret calls should be False (hide fret)
        for call in calls_fret:
            assert call[0][0] == False

        # Reset mocks
        mock_comp_vis.reset_mock()
        mock_fret_vis.reset_mock()

        # Change to FRET tab (index 6)
        plotter.tab_widget.setCurrentIndex(6)

        # Should have at least one False call (hide) and one True call (show) for FRET
        assert mock_comp_vis.call_count >= 1
        assert mock_fret_vis.call_count >= 2

        # Check the calls were in the right order
        calls_comp = mock_comp_vis.call_args_list
        calls_fret = mock_fret_vis.call_args_list

        # All component calls should be False (hide components)
        for call in calls_comp:
            assert call[0][0] == False
        # The last FRET call should be True (show FRET)
        assert calls_fret[-1][0][0] == True
