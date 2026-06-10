import contextlib
import warnings
from unittest.mock import patch

import numpy as np
import pytest
from biaplotter.plotter import CanvasWidget
from napari.layers import Image
from qtpy.QtWidgets import (
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
)

from napari_phasors._synthetic_generator import (
    make_intensity_layer_with_phasors,
    make_raw_flim_data,
)
from napari_phasors.calibration_tab import CalibrationWidget
from napari_phasors.components_tab import ComponentsWidget
from napari_phasors.filter_tab import FilterWidget
from napari_phasors.fret_tab import FretWidget
from napari_phasors.plotter import (
    PlotterWidget,
)
from napari_phasors.selection_tab import SelectionWidget


def create_image_layer_with_phasors(harmonic=None):
    """Create an intensity image layer with phasors for testing."""
    if harmonic is None:
        harmonic = [1, 2, 3]
    time_constants = [0.1, 1, 2, 3, 4, 5, 10]
    raw_flim_data = make_raw_flim_data(time_constants=time_constants)
    harmonic = harmonic
    return make_intensity_layer_with_phasors(raw_flim_data, harmonic=harmonic)


def test_nap_plot_tools_safe_disconnect_patch_is_applied_and_idempotent():
    """The runtime patch should be installed once and remain safe to re-run."""
    pytest.importorskip("nap_plot_tools")

    from nap_plot_tools.tools import CustomToolbarWidget

    from napari_phasors import plotter as plotter_module

    assert getattr(
        CustomToolbarWidget,
        '_napari_phasors_safe_disconnect_patch',
        False,
    )

    # Reapplying should be a no-op (idempotent guard path).
    plotter_module._patch_nap_plot_tools_safe_disconnect()
    plotter_module._patch_nap_plot_tools_safe_disconnect()

    assert getattr(
        CustomToolbarWidget,
        '_napari_phasors_safe_disconnect_patch',
        False,
    )


def test_nap_plot_tools_safe_disconnect_rewires_callbacks_without_warning(
    qtbot,
):
    """Rewiring toolbar callbacks should not emit Qt disconnect RuntimeWarning."""
    pytest.importorskip("nap_plot_tools")

    from nap_plot_tools.tools import CustomToolbarWidget

    toolbar = CustomToolbarWidget()
    qtbot.addWidget(toolbar)

    toggle_calls = []
    click_calls = []

    def toggle_cb_1(checked):
        toggle_calls.append(('toggle_cb_1', bool(checked)))

    def toggle_cb_2(checked):
        toggle_calls.append(('toggle_cb_2', bool(checked)))

    def click_cb_1():
        click_calls.append('click_cb_1')

    def click_cb_2():
        click_calls.append('click_cb_2')

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')

        # Checkable path (toggled signal).
        toolbar.add_custom_button(
            name='SEL',
            default_icon_path='dummy_default.png',
            checked_icon_path='dummy_checked.png',
            callback=toggle_cb_1,
        )
        toolbar.buttons['SEL'].setChecked(True)
        toolbar.connect_button_callback('SEL', toggle_cb_2)
        toolbar.buttons['SEL'].setChecked(False)
        toolbar.connect_button_callback('SEL', None)
        toolbar.buttons['SEL'].setChecked(True)

        # Non-checkable path (clicked signal).
        toolbar.add_custom_button(
            name='ACT',
            default_icon_path='dummy_action.png',
            callback=click_cb_1,
        )
        toolbar.buttons['ACT'].click()
        toolbar.connect_button_callback('ACT', click_cb_2)
        toolbar.buttons['ACT'].click()
        toolbar.connect_button_callback('ACT', None)
        toolbar.buttons['ACT'].click()

        # Unknown button name should be safely ignored.
        toolbar.connect_button_callback('MISSING', toggle_cb_1)

    assert ('toggle_cb_1', True) in toggle_calls
    assert ('toggle_cb_2', False) in toggle_calls
    assert all(name != 'toggle_cb_1' for name, _ in toggle_calls[1:])

    assert click_calls[:2] == ['click_cb_1', 'click_cb_2']
    assert click_calls.count('click_cb_1') == 1
    assert click_calls.count('click_cb_2') == 1

    disconnect_warnings = [
        w
        for w in caught
        if 'Failed to disconnect' in str(w.message)
        and 'toggled(bool)' in str(w.message)
    ]
    assert not disconnect_warnings


def test_biaplotter_toggle_callback_handles_missing_sender(
    make_viewer_model,
):
    """Pan/zoom toggles emitted via non-Qt signals must not rely on sender()."""
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)
    canvas = plotter.canvas_widget

    canvas.toolbar.mode = 'pan/zoom'

    with (
        patch.object(
            canvas, '_deactivate_and_remove_all_selectors'
        ) as mock_deactivate,
        patch.object(CanvasWidget, 'sender', return_value=None),
    ):
        canvas._on_toggle_button(True)

    mock_deactivate.assert_called_once()


def test_biaplotter_pan_toggle_deactivates_active_selector_without_sender(
    make_viewer_model,
):
    """A sender-less pan toggle should clear the current selector tool."""
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)
    canvas = plotter.canvas_widget

    with patch.object(CanvasWidget, 'sender', return_value=None):
        canvas.selection_toolbar.buttons['LASSO'].click()

    canvas.toolbar.mode = 'pan/zoom'

    with patch.object(CanvasWidget, 'sender', return_value=None):
        canvas._on_toggle_button(True)

    assert canvas.active_selector is None
    assert all(
        not button.isChecked()
        for button in canvas.selection_toolbar.buttons.values()
    )


def test_biaplotter_selector_toggle_handles_missing_sender(
    make_viewer_model,
):
    """Selector buttons should still activate when sender() is unavailable."""
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)
    canvas = plotter.canvas_widget

    with patch.object(CanvasWidget, 'sender', return_value=None):
        canvas.selection_toolbar.buttons['LASSO'].click()

    assert canvas.active_selector is not None
    assert (
        canvas.active_selector.__class__.__name__ == 'InteractiveLassoSelector'
    )


def test_biaplotter_selector_toggle_is_exclusive_without_sender(
    make_viewer_model,
):
    """Switching selector tools without sender() should deactivate the previous tool."""
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)
    canvas = plotter.canvas_widget

    with patch.object(CanvasWidget, 'sender', return_value=None):
        canvas.selection_toolbar.buttons['LASSO'].click()
        canvas.selection_toolbar.buttons['ELLIPSE'].click()

    assert canvas.selection_toolbar.buttons['LASSO'].isChecked() is False
    assert canvas.selection_toolbar.buttons['ELLIPSE'].isChecked() is True
    assert canvas.active_selector is not None
    assert (
        canvas.active_selector.__class__.__name__
        == 'InteractiveEllipseSelector'
    )


def test_biaplotter_zoom_toggle_deactivates_active_selector_without_sender(
    make_viewer_model,
):
    """A sender-less zoom toggle should clear the current selector tool."""
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)
    canvas = plotter.canvas_widget

    with patch.object(CanvasWidget, 'sender', return_value=None):
        canvas.selection_toolbar.buttons['RECTANGLE'].click()

    canvas.toolbar.mode = 'zoom rect'

    with patch.object(CanvasWidget, 'sender', return_value=None):
        canvas._on_toggle_button(True)

    assert canvas.active_selector is None
    assert all(
        not button.isChecked()
        for button in canvas.selection_toolbar.buttons.values()
    )


def test_phasor_plotter_initialization_values(make_viewer_model):
    """Test the initialization of the Phasor Plotter Widget."""
    viewer = make_viewer_model()

    # Create Plotter widget
    plotter = PlotterWidget(viewer)

    # Basic widget structure tests
    assert plotter.viewer == viewer
    assert isinstance(plotter.layout(), QVBoxLayout)

    # Canvas widget tests
    assert hasattr(plotter, 'canvas_widget')
    assert isinstance(plotter.canvas_widget, CanvasWidget)
    assert plotter.canvas_widget.minimumSize().width() >= 300
    assert plotter.canvas_widget.minimumSize().height() >= 300
    assert plotter.canvas_widget.class_spinbox.value == 1

    # UI components tests
    assert hasattr(plotter, 'image_layer_with_phasor_features_combobox')
    # Note: image_layer_with_phasor_features_combobox is now a wrapper, not QComboBox
    assert hasattr(plotter, 'harmonic_spinbox')
    assert isinstance(plotter.harmonic_spinbox, QSpinBox)
    assert plotter.harmonic_spinbox.minimum() == 1
    assert plotter.harmonic_spinbox.value() == 1

    # Import buttons tests
    assert hasattr(plotter, 'import_from_layer_button')
    assert isinstance(plotter.import_from_layer_button, QPushButton)
    assert plotter.import_from_layer_button.text() == "Layer"

    assert hasattr(plotter, 'import_from_file_button')
    assert isinstance(plotter.import_from_file_button, QPushButton)
    assert plotter.import_from_file_button.text() == "OME-TIFF File"

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
        "Filter",
        "Selection",
        "Components",
        "Phasor Mapping",
        "FRET",
    ]
    assert tab_names == expected_tabs

    # Test individual tabs exist
    assert hasattr(plotter, 'settings_tab')
    assert hasattr(plotter, 'calibration_tab')
    assert hasattr(plotter, 'filter_tab')
    assert hasattr(plotter, 'selection_tab')
    assert hasattr(plotter, 'components_tab')
    assert hasattr(plotter, 'phasor_mapping_tab')
    assert hasattr(plotter, 'fret_tab')

    # Test filter_tab and selection_tab are proper widgets
    assert isinstance(plotter.filter_tab, FilterWidget)
    assert isinstance(plotter.selection_tab, SelectionWidget)
    assert isinstance(plotter.calibration_tab, CalibrationWidget)
    assert isinstance(plotter.components_tab, ComponentsWidget)
    assert isinstance(plotter.fret_tab, FretWidget)

    # Test settings tab inputs widget
    assert hasattr(plotter, 'plotter_inputs_widget')
    assert hasattr(plotter.plotter_inputs_widget, 'plot_type_combobox')
    assert hasattr(plotter.plotter_inputs_widget, 'colormap_combobox')
    assert hasattr(plotter.plotter_inputs_widget, 'number_of_bins_spinbox')
    assert hasattr(plotter.plotter_inputs_widget, 'semi_circle_checkbox')
    assert hasattr(plotter.plotter_inputs_widget, 'white_background_checkbox')
    assert hasattr(plotter.plotter_inputs_widget, 'log_scale_checkbox')
    assert hasattr(plotter.plotter_inputs_widget, 'marker_size_spinbox')
    assert hasattr(plotter.plotter_inputs_widget, 'marker_alpha_spinbox')
    assert hasattr(plotter.plotter_inputs_widget, 'marker_color_button')

    # Test default property values
    assert plotter.harmonic == 1
    assert plotter.plot_type == 'HISTOGRAM2D'
    assert plotter.histogram_colormap == 'turbo'
    assert plotter.toggle_semi_circle  # Should default to semi-circle mode
    assert plotter.white_background  # Should default to white background

    # Test plot type combobox items
    plot_types = []
    for i in range(plotter.plotter_inputs_widget.plot_type_combobox.count()):
        plot_types.append(
            plotter.plotter_inputs_widget.plot_type_combobox.itemText(i)
        )
    assert plot_types == [
        "Density Plot (2D Histogram)",
        "Dot Plot (Scatter)",
        "Contour Plot",
        "None",
    ]

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
    assert xlim[0] == -0.1 and xlim[1] == 1.1
    assert ylim[0] == -0.1 and ylim[1] == 0.7


def test_phasor_plotter_initialization_plot_not_called(make_viewer_model):
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)

    # Patch after widget construction to avoid Qt teardown issues from
    # constructing PlotterWidget inside a patch context manager.
    with patch.object(plotter, '_set_active_artist_and_plot') as mock_plot:
        # Check that modifying values does not call plot
        plotter.plot_type = 'SCATTER'
        mock_plot.assert_not_called()
        plotter.histogram_colormap = 'viridis'
        mock_plot.assert_not_called()
        plotter.toggle_semi_circle = False
        mock_plot.assert_not_called()
        plotter.white_background = False
        mock_plot.assert_not_called()
        plotter.plotter_inputs_widget.semi_circle_checkbox.setChecked(True)
        mock_plot.assert_not_called()
        plotter.plotter_inputs_widget.white_background_checkbox.setChecked(
            False
        )
        mock_plot.assert_not_called()

    plotter.deleteLater()


def test_phasor_plotter_initialization_with_layer(make_viewer_model):
    """Test the initialization of the Phasor Plotter Widget with a layer already present."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    plotter = PlotterWidget(viewer)
    # Test that the layer is automatically detected and selected
    assert plotter.image_layers_checkable_combobox.count() == 1
    assert (
        plotter.image_layer_with_phasor_features_combobox.currentText()
        == intensity_image_layer.name
    )

    # Test that the phasor data is available in metadata
    assert "G" in intensity_image_layer.metadata
    assert "S" in intensity_image_layer.metadata
    assert "harmonics" in intensity_image_layer.metadata

    # Test harmonic spinbox maximum is set based on data
    harmonics = np.atleast_1d(intensity_image_layer.metadata["harmonics"])
    expected_max_harmonic = int(np.max(harmonics))
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

    plotter.deleteLater()


def test_adding_removing_layers_updates_plot(make_viewer_model):
    """Test that adding/removing layers updates the plotter widget."""
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)

    with patch.object(plotter, '_set_active_artist_and_plot') as mock_plot:
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
        plotter.plotter_inputs_widget.semi_circle_checkbox.setChecked(True)
        mock_plot.assert_not_called()
        plotter.plotter_inputs_widget.white_background_checkbox.setChecked(
            False
        )
        mock_plot.assert_not_called()

        # Add two layers with phasor features
        intensity_image_layer_2 = create_image_layer_with_phasors()
        viewer.add_layer(intensity_image_layer_2)
        # Plot can be refreshed more than once depending on signal order.
        assert mock_plot.call_count >= 1
        mock_plot.reset_mock()  # Reset mock for next call

        viewer.add_layer(intensity_image_layer)
        mock_plot.assert_not_called()

        # Check values were not reset to defaults when new layer was added
        assert plotter.plot_type == 'SCATTER'
        assert plotter.histogram_colormap == 'viridis'
        assert not plotter.toggle_semi_circle
        assert not plotter.white_background
        assert plotter.plotter_inputs_widget.semi_circle_checkbox.isChecked()
        assert (
            not plotter.plotter_inputs_widget.white_background_checkbox.isChecked()
        )

        # Check that the combobox has both layers with phasor features
        assert plotter.image_layers_checkable_combobox.count() == 2
        # Check that the first layer is still selected
        assert (
            plotter.image_layer_with_phasor_features_combobox.currentText()
            == intensity_image_layer_2.name
        )

    plotter.deleteLater()


def test_layer_settings_persistence_across_layer_switches(make_viewer_model):
    """Test that settings persist when switching between layers."""
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)

    # Create two layers
    layer_1 = create_image_layer_with_phasors()
    layer_2 = create_image_layer_with_phasors()
    viewer.add_layer(layer_1)
    viewer.add_layer(layer_2)

    # Set layer_1 as active
    plotter.image_layer_with_phasor_features_combobox.setCurrentText(
        layer_1.name
    )

    # Modify settings for layer_1
    plotter.plot_type = 'SCATTER'
    plotter.histogram_colormap = 'viridis'
    plotter.toggle_semi_circle = False
    plotter.white_background = False

    # Switch to layer_2 (should have defaults)
    plotter.image_layer_with_phasor_features_combobox.setCurrentText(
        layer_2.name
    )
    assert plotter.plot_type == 'SCATTER'
    assert plotter.histogram_colormap == 'viridis'
    assert not plotter.toggle_semi_circle
    assert not plotter.white_background

    # Switch back to layer_1 (should restore settings)
    plotter.image_layer_with_phasor_features_combobox.setCurrentText(
        layer_1.name
    )
    assert plotter.plot_type == 'SCATTER'
    assert plotter.histogram_colormap == 'viridis'
    assert not plotter.toggle_semi_circle
    assert not plotter.white_background

    plotter.deleteLater()


def test_layer_settings_initialization_in_metadata(make_viewer_model):
    """Test that settings are properly initialized in layer metadata."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)
    plotter = PlotterWidget(viewer)

    # Check that settings were initialized
    assert 'settings' in layer.metadata
    assert 'harmonic' in layer.metadata['settings']
    assert 'semi_circle' in layer.metadata['settings']
    assert 'white_background' in layer.metadata['settings']
    assert 'plot_type' in layer.metadata['settings']
    assert 'colormap' in layer.metadata['settings']
    assert 'number_of_bins' in layer.metadata['settings']
    assert 'log_scale' in layer.metadata['settings']
    assert 'marker_size' in layer.metadata['settings']
    assert 'marker_alpha' in layer.metadata['settings']
    assert 'marker_color' in layer.metadata['settings']

    # Check default values
    assert layer.metadata['settings']['harmonic'] == 1
    assert layer.metadata['settings']['semi_circle']
    assert layer.metadata['settings']['white_background']
    assert layer.metadata['settings']['plot_type'] == 'HISTOGRAM2D'
    assert layer.metadata['settings']['colormap'] == 'turbo'
    assert plotter.plotter_inputs_widget.marker_size_spinbox.value() == 50
    assert layer.metadata['settings']['number_of_bins'] == 150
    assert not layer.metadata['settings']['log_scale']
    assert layer.metadata['settings']['marker_size'] == 50
    assert layer.metadata['settings']['marker_alpha'] == 0.5
    assert layer.metadata['settings']['marker_color'] == '#1f77b4'

    plotter.deleteLater()


def test_layer_settings_update_in_metadata(make_viewer_model):
    """Test that changing settings updates layer metadata."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)
    plotter = PlotterWidget(viewer)

    # Change settings
    plotter.harmonic = 2
    plotter.plot_type = 'SCATTER'
    plotter.histogram_colormap = 'viridis'
    plotter.toggle_semi_circle = False
    plotter.white_background = False

    # Verify metadata was updated
    assert layer.metadata['settings']['harmonic'] == 2
    assert layer.metadata['settings']['plot_type'] == 'SCATTER'
    assert layer.metadata['settings']['colormap'] == 'viridis'
    assert not layer.metadata['settings']['semi_circle']
    assert not layer.metadata['settings']['white_background']

    plotter.deleteLater()


def test_adding_layer_without_settings_initializes_defaults(
    make_viewer_model,
):
    """Test that adding a layer without settings metadata initializes defaults."""
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)

    # Create a layer and remove settings if they exist
    layer = create_image_layer_with_phasors()
    if 'settings' in layer.metadata:
        del layer.metadata['settings']

    viewer.add_layer(layer)

    # Verify settings were initialized with defaults
    assert 'settings' in layer.metadata
    assert layer.metadata['settings']['harmonic'] == 1
    assert layer.metadata['settings']['semi_circle']
    assert layer.metadata['settings']['white_background']
    assert layer.metadata['settings']['plot_type'] == 'HISTOGRAM2D'

    plotter.deleteLater()


def test_modifying_settings_updates_metadata_correctly(make_viewer_model):
    """Test that each setting change correctly updates metadata."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)
    plotter = PlotterWidget(viewer)

    # Test harmonic update
    plotter.harmonic = 3
    assert layer.metadata['settings']['harmonic'] == 3

    # Test plot type update
    plotter.plotter_inputs_widget.plot_type_combobox.setCurrentText(
        "Dot Plot (Scatter)"
    )
    assert layer.metadata['settings']['plot_type'] == 'SCATTER'

    # Test colormap update
    plotter.plotter_inputs_widget.colormap_combobox.setCurrentText('viridis')
    assert layer.metadata['settings']['colormap'] == 'viridis'

    # Test bins update
    plotter.plotter_inputs_widget.number_of_bins_spinbox.setValue(200)
    assert layer.metadata['settings']['number_of_bins'] == 200

    # Test log scale update
    plotter.plotter_inputs_widget.log_scale_checkbox.setChecked(True)
    assert layer.metadata['settings']['log_scale']

    # Test semi circle update
    plotter.plotter_inputs_widget.semi_circle_checkbox.setChecked(True)
    assert not layer.metadata['settings']['semi_circle']

    # Test white background update
    plotter.plotter_inputs_widget.white_background_checkbox.setChecked(False)
    assert not layer.metadata['settings']['white_background']

    # Test marker size update
    plotter.plotter_inputs_widget.marker_size_spinbox.setValue(30)
    assert layer.metadata['settings']['marker_size'] == 30

    # Test marker alpha update
    plotter.plotter_inputs_widget.marker_alpha_spinbox.setValue(0.8)
    assert layer.metadata['settings']['marker_alpha'] == 0.8

    # Test marker color update
    plotter._marker_color = '#ff0000'
    plotter._update_setting_in_metadata('marker_color', '#ff0000')
    assert layer.metadata['settings']['marker_color'] == '#ff0000'

    plotter.deleteLater()


def test_plot_type_ui_toggles(make_viewer_model):
    """Test that switching plot types hides and shows correct inputs."""
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)

    # Initial is HISTOGRAM2D
    assert plotter.plot_type == 'HISTOGRAM2D'
    assert not plotter._colormap_row_widget.isHidden()
    assert not plotter.plotter_inputs_widget.number_of_bins_spinbox.isHidden()
    assert not plotter.plotter_inputs_widget.log_scale_checkbox.isHidden()

    assert plotter.plotter_inputs_widget.marker_size_spinbox.isHidden()
    assert plotter.plotter_inputs_widget.marker_alpha_spinbox.isHidden()
    assert plotter.plotter_inputs_widget.marker_color_button.isHidden()

    # Change to SCATTER
    plotter.plotter_inputs_widget.plot_type_combobox.setCurrentText(
        "Dot Plot (Scatter)"
    )

    assert plotter._colormap_row_widget.isHidden()
    assert plotter.plotter_inputs_widget.number_of_bins_spinbox.isHidden()
    assert plotter.plotter_inputs_widget.log_scale_checkbox.isHidden()

    assert not plotter.plotter_inputs_widget.marker_size_spinbox.isHidden()
    assert not plotter.plotter_inputs_widget.marker_alpha_spinbox.isHidden()
    assert not plotter.plotter_inputs_widget.marker_color_button.isHidden()

    # Change to CONTOUR
    plotter.plotter_inputs_widget.plot_type_combobox.setCurrentText(
        "Contour Plot"
    )

    assert not plotter._colormap_row_widget.isHidden()
    assert not plotter.plotter_inputs_widget.number_of_bins_spinbox.isHidden()
    assert plotter.plotter_inputs_widget.log_scale_checkbox.isHidden()

    assert plotter.plotter_inputs_widget.marker_size_spinbox.isHidden()
    assert plotter.plotter_inputs_widget.marker_alpha_spinbox.isHidden()
    assert plotter.plotter_inputs_widget.marker_color_button.isHidden()

    assert not plotter.plotter_inputs_widget.contour_levels_spinbox.isHidden()
    assert (
        not plotter.plotter_inputs_widget.contour_linewidth_spinbox.isHidden()
    )

    # Change to None
    plotter.plotter_inputs_widget.plot_type_combobox.setCurrentText("None")

    assert plotter._colormap_row_widget.isHidden()
    assert plotter.plotter_inputs_widget.number_of_bins_spinbox.isHidden()
    assert plotter.plotter_inputs_widget.log_scale_checkbox.isHidden()

    assert plotter.plotter_inputs_widget.marker_size_spinbox.isHidden()
    assert plotter.plotter_inputs_widget.marker_alpha_spinbox.isHidden()
    assert plotter.plotter_inputs_widget.marker_color_button.isHidden()

    assert plotter.plotter_inputs_widget.contour_levels_spinbox.isHidden()
    assert plotter.plotter_inputs_widget.contour_linewidth_spinbox.isHidden()

    plotter.deleteLater()


def test_contour_plot_creates_and_cleans_collections(make_viewer_model):
    """Contour mode should create contour artists and clean them when mode changes."""
    viewer = make_viewer_model()
    layer1 = create_image_layer_with_phasors()
    layer2 = create_image_layer_with_phasors()
    viewer.add_layer(layer1)
    viewer.add_layer(layer2)
    plotter = PlotterWidget(viewer)

    def contour_artists_in_axes():
        return [
            artist
            for artist in plotter.canvas_widget.axes.collections
            if artist.get_label() == 'contour_plot_element'
        ]

    # Single-layer contour render with real phasor data.
    plotter.image_layers_checkable_combobox.setCheckedItems([layer1.name])
    plotter._process_layer_selection_change()
    plotter.plot_type = 'CONTOUR'
    plotter.plot()

    assert len(plotter._contour_collections) > 0
    assert len(contour_artists_in_axes()) > 0

    # Multi-layer contour render should also succeed without exceptions.
    plotter.image_layers_checkable_combobox.setCheckedItems(
        [layer1.name, layer2.name]
    )
    plotter._process_layer_selection_change()
    plotter.plot_type = 'CONTOUR'
    plotter.plot()

    assert len(plotter._contour_collections) > 0
    assert len(contour_artists_in_axes()) > 0

    # Switching away from contour should remove tracked collections.
    plotter.plot_type = 'SCATTER'
    plotter.plot()

    assert plotter._contour_collections == []
    assert len(contour_artists_in_axes()) == 0

    # Switching back recreates contour artists.
    plotter.plot_type = 'CONTOUR'
    plotter.plot()

    assert len(plotter._contour_collections) > 0
    assert len(contour_artists_in_axes()) > 0

    plotter.deleteLater()


def test_contour_plot_grouped_mode(make_viewer_model):
    """Test that grouped mode creates one contour collection per group."""
    viewer = make_viewer_model()
    layer1 = create_image_layer_with_phasors()
    layer2 = create_image_layer_with_phasors()
    viewer.add_layer(layer1)
    viewer.add_layer(layer2)
    plotter = PlotterWidget(viewer)

    plotter.image_layers_checkable_combobox.setCheckedItems(
        [layer1.name, layer2.name]
    )
    plotter._process_layer_selection_change()

    # Configure grouped mode
    plotter.plot_type = 'CONTOUR'
    plotter._contour_display_mode = "Grouped"
    plotter._contour_group_assignments = {layer1.name: 1, layer2.name: 1}
    plotter._contour_group_names = {1: "Group 1"}

    plotter.plot()

    # Should only create one contour collection for the entire group
    assert len(plotter._contour_collections) == 1

    artists = [
        artist
        for artist in plotter.canvas_widget.axes.collections
        if artist.get_label() == 'contour_plot_element'
    ]
    assert len(artists) > 0

    plotter.deleteLater()


def test_make_solid_contour_cmap():
    """Test that the solid contour colormap blends the target color with 50% white."""
    # We don't need a full widget, just the method
    from napari_phasors.plotter import PlotterWidget

    # Create an instance without initializing a viewer (or use a mock)
    # The method _make_solid_contour_cmap doesn't use instance state.
    # We'll just call it on an uninitialized instance or dummy instance.
    class DummyPlotter:
        _normalize_rgb = staticmethod(PlotterWidget._normalize_rgb)
        _make_solid_contour_cmap = PlotterWidget._make_solid_contour_cmap

    dummy = DummyPlotter()

    target_color = (1.0, 0.0, 0.0)  # Red
    cmap = dummy._make_solid_contour_cmap("test_cmap", target_color)

    # Low color should be 50% white blended with red
    # low_color = np.clip([1, 0, 0] + (1 - [1, 0, 0]) * 0.5, 0, 1) = [1, 0.5, 0.5]
    np.testing.assert_allclose(cmap(0.0)[:3], (1.0, 0.5, 0.5))

    # High color should be the target color
    np.testing.assert_allclose(cmap(1.0)[:3], target_color)


def test_add_layer_without_phasor_features_does_not_trigger_plot_or_combobox(
    make_viewer_model,
):
    """Test that adding a layer without phasor features does not call plot or update the combobox."""
    from napari.layers import Image

    viewer = make_viewer_model()

    plotter = PlotterWidget(viewer)

    # Patch plot and on_image_layer_changed after widget construction
    with (
        patch.object(plotter, '_set_active_artist_and_plot') as mock_plot,
        patch.object(plotter, 'on_image_layer_changed') as mock_layer_changed,
    ):
        # Add a regular image layer (no phasor features)
        regular_layer = Image(np.random.random((10, 10)))
        viewer.add_layer(regular_layer)

        # plot and on_image_layer_changed should NOT be called
        mock_plot.assert_not_called()
        mock_layer_changed.assert_not_called()

        # The combobox should not be updated
        assert plotter.image_layers_checkable_combobox.count() == 0
        assert (
            plotter.image_layer_with_phasor_features_combobox.currentText()
            == ''
        )

    plotter.deleteLater()


def test_phasor_plotter_property_setters(make_viewer_model):
    """Test property setters of the Phasor Plotter Widget."""
    viewer = make_viewer_model()
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
    assert plotter.white_background
    assert plotter.plotter_inputs_widget.white_background_checkbox.isChecked()

    # Test toggle_semi_circle setter
    plotter.toggle_semi_circle = False
    assert not plotter.toggle_semi_circle
    assert plotter.plotter_inputs_widget.semi_circle_checkbox.isChecked()


def test_phasor_plotter_layer_management(make_viewer_model):
    """Test layer management functionality of the Phasor Plotter Widget."""
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)

    # Initially no layers with phasor features
    assert plotter.image_layers_checkable_combobox.count() == 0

    # Add layer with phasor features
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)

    # Check that combobox is updated
    assert plotter.image_layers_checkable_combobox.count() == 1
    assert (
        plotter.image_layer_with_phasor_features_combobox.currentText()
        == intensity_image_layer.name
    )

    # Add another layer without phasor features
    regular_layer = Image(np.random.random((10, 10)))
    viewer.add_layer(regular_layer)

    # Combobox should still have only the layer with phasor features
    assert plotter.image_layers_checkable_combobox.count() == 1

    # Remove the phasor layer
    viewer.layers.remove(intensity_image_layer)

    # Combobox should be empty again
    assert plotter.image_layers_checkable_combobox.count() == 0


def test_phasor_plotter_semicircle_checkbox(make_viewer_model):
    """Test semicircle checkbox functionality."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    plotter = PlotterWidget(viewer)

    # Initially semicircle should be enabled (default)
    assert plotter.toggle_semi_circle
    assert not plotter.plotter_inputs_widget.semi_circle_checkbox.isChecked()

    # Get initial axes limits (semicircle mode)
    initial_ylim = plotter.canvas_widget.axes.get_ylim()

    # Semicircle should have y-limits starting from 0 or slightly below
    assert initial_ylim[0] <= 0.1  # Small tolerance for padding

    # Check full-circle toggle (should show full circle)
    plotter.plotter_inputs_widget.semi_circle_checkbox.setChecked(True)
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
    plotter.plotter_inputs_widget.semi_circle_checkbox.setChecked(False)
    plotter.toggle_semi_circle = True

    # Should return to semicircle limits
    restored_ylim = plotter.canvas_widget.axes.get_ylim()
    assert restored_ylim[0] >= new_ylim[0]  # Should not extend as far down


def test_phasor_plotter_colormap_combobox(make_viewer_model):
    """Test colormap combobox functionality."""
    viewer = make_viewer_model()
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
        if colormap_name not in {initial_colormap, 'Select color...'}:
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


def test_phasor_plotter_log_scale_checkbox(make_viewer_model):
    """Test log scale checkbox functionality."""
    viewer = make_viewer_model()
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


def test_density_plot_log_scale_warning_suppression(make_viewer_model):
    """Test that warnings related to log normalization and non-positive ylim are suppressed in HISTOGRAM2D mode."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    plotter = PlotterWidget(viewer)

    # Enable Density Plot mode and log scale
    plotter.plot_type = 'HISTOGRAM2D'
    plotter.histogram_log_scale = True

    with warnings.catch_warnings(record=True) as caught_warnings:
        # Trigger plot update while log scale is active
        plotter.plot()

        # Verify that none of the suppressed warnings were raised
        for w in caught_warnings:
            msg = str(w.message)
            if (
                "Log normalization applied" in msg
                or "non-positive ylim" in msg
            ):
                pytest.fail(f"Warning was not suppressed: {msg}")


def test_phasor_plotter_white_background_checkbox(make_viewer_model):
    """Test white background checkbox functionality."""
    viewer = make_viewer_model()
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


def test_phasor_plotter_colorbar_updates(make_viewer_model):
    """Test that colorbar updates when histogram settings change."""
    viewer = make_viewer_model()
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


def test_on_image_layer_changed_prevents_recursion(
    make_viewer_model,
):
    """Test that the method prevents recursive calls using the guard flag."""

    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    plotter = PlotterWidget(viewer)

    # Set the guard flag to simulate being already in the method
    plotter._in_on_image_layer_changed = True

    with patch.object(plotter, 'plot') as mock_plot:
        # Call the method - should return early due to guard
        plotter.on_image_layer_changed()

        # Verify plot was not called due to guard
        mock_plot.assert_not_called()
        plotter.deleteLater()

    # Verify guard flag is still True (not reset by early return)
    assert plotter._in_on_image_layer_changed


def test_on_image_layer_changed_with_empty_layer_name(
    make_viewer_model,
):
    """Test behavior when combobox has empty layer name."""

    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)

    # Ensure combobox is empty by deselecting all
    plotter.image_layers_checkable_combobox.setCheckedItems([])

    with patch.object(plotter, 'plot') as mock_plot:
        plotter.on_image_layer_changed()

        # Should not call plot when layer name is empty
        mock_plot.assert_not_called()
        plotter.deleteLater()

    # Verify no layer is selected when combobox is empty
    assert (
        plotter.image_layer_with_phasor_features_combobox.currentText() == ''
    )

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


def test_on_image_layer_changed_sets_harmonic_maximum(
    make_viewer_model,
):
    """Test that harmonic spinbox maximum is set based on layer data."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    plotter = PlotterWidget(viewer)

    # Get expected maximum harmonic from the data using new array-based structure
    harmonics = np.atleast_1d(intensity_image_layer.metadata["harmonics"])
    expected_max = int(np.max(harmonics))

    # Call the method
    plotter.on_image_layer_changed()

    # Verify harmonic spinbox maximum is set correctly
    assert plotter.harmonic_spinbox.maximum() == expected_max


def test_on_image_layer_changed_updates_phasors_selected_layer(
    make_viewer_model,
):
    """Test that the phasor data is accessible after layer change."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    plotter = PlotterWidget(viewer)

    # Verify the selected layer name is in the combobox
    layer_name = (
        plotter.image_layer_with_phasor_features_combobox.currentText()
    )
    assert layer_name == intensity_image_layer.name

    # Verify the layer has phasor data
    selected_layer = viewer.layers[layer_name]
    assert "G" in selected_layer.metadata
    assert "S" in selected_layer.metadata
    assert "harmonics" in selected_layer.metadata

    # Call the method
    plotter.on_image_layer_changed()

    # Verify phasor data is still accessible
    layer_name = (
        plotter.image_layer_with_phasor_features_combobox.currentText()
    )
    assert layer_name == intensity_image_layer.name
    selected_layer = viewer.layers[layer_name]
    assert "G" in selected_layer.metadata
    assert "S" in selected_layer.metadata


def test_on_image_layer_changed_guard_flag_cleanup(
    make_viewer_model,
):
    """Test that guard flag is properly cleaned up even if an exception occurs."""

    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    plotter = PlotterWidget(viewer)

    # Mock plot to raise an exception
    with patch.object(
        plotter, 'plot', side_effect=Exception("Test exception")
    ):
        with contextlib.suppress(Exception):
            plotter.on_image_layer_changed()
        plotter.deleteLater()

    # Verify guard flag is cleaned up even after exception
    assert (
        not hasattr(plotter, '_in_on_image_layer_changed')
        or not plotter._in_on_image_layer_changed
    )


def test_on_image_layer_changed_multiple_calls(
    make_viewer_model,
):
    """Test multiple sequential calls to ensure plot is called each time."""

    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    plotter = PlotterWidget(viewer)

    with patch.object(plotter, 'plot') as mock_plot:
        # Call multiple times
        plotter.on_image_layer_changed()
        plotter.on_image_layer_changed()
        plotter.on_image_layer_changed()

        # Each call should result in plot being called once
        assert mock_plot.call_count == 3
        plotter.deleteLater()
