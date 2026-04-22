import contextlib
import warnings
from unittest.mock import patch

import numpy as np
from biaplotter.plotter import CanvasWidget
from napari.layers import Image
from qtpy.QtWidgets import (
    QComboBox,
    QLabel,
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
    PhasorCenterLayerSettingsDialog,
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


def test_phasor_plotter_initialization_values(make_napari_viewer):
    """Test the initialization of the Phasor Plotter Widget."""
    viewer = make_napari_viewer()

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


def test_phasor_plotter_initialization_plot_not_called(make_napari_viewer):
    viewer = make_napari_viewer()
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


def test_phasor_plotter_initialization_with_layer(make_napari_viewer):
    """Test the initialization of the Phasor Plotter Widget with a layer already present."""
    viewer = make_napari_viewer()
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


def test_adding_removing_layers_updates_plot(make_napari_viewer):
    """Test that adding/removing layers updates the plotter widget."""
    viewer = make_napari_viewer()
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


def test_layer_settings_persistence_across_layer_switches(make_napari_viewer):
    """Test that settings persist when switching between layers."""
    viewer = make_napari_viewer()
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


def test_layer_settings_initialization_in_metadata(make_napari_viewer):
    """Test that settings are properly initialized in layer metadata."""
    viewer = make_napari_viewer()
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


def test_layer_settings_update_in_metadata(make_napari_viewer):
    """Test that changing settings updates layer metadata."""
    viewer = make_napari_viewer()
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
    make_napari_viewer,
):
    """Test that adding a layer without settings metadata initializes defaults."""
    viewer = make_napari_viewer()
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


def test_modifying_settings_updates_metadata_correctly(make_napari_viewer):
    """Test that each setting change correctly updates metadata."""
    viewer = make_napari_viewer()
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


def test_plot_type_ui_toggles(make_napari_viewer):
    """Test that switching plot types hides and shows correct inputs."""
    viewer = make_napari_viewer()
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


def test_contour_plot_creates_and_cleans_collections(make_napari_viewer):
    """Contour mode should create contour artists and clean them when mode changes."""
    viewer = make_napari_viewer()
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


def test_add_layer_without_phasor_features_does_not_trigger_plot_or_combobox(
    make_napari_viewer,
):
    """Test that adding a layer without phasor features does not call plot or update the combobox."""
    from napari.layers import Image

    viewer = make_napari_viewer()

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
    assert plotter.white_background
    assert plotter.plotter_inputs_widget.white_background_checkbox.isChecked()

    # Test toggle_semi_circle setter
    plotter.toggle_semi_circle = False
    assert not plotter.toggle_semi_circle
    assert plotter.plotter_inputs_widget.semi_circle_checkbox.isChecked()


def test_phasor_plotter_layer_management(make_napari_viewer):
    """Test layer management functionality of the Phasor Plotter Widget."""
    viewer = make_napari_viewer()
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


def test_phasor_plotter_semicircle_checkbox(make_napari_viewer):
    """Test semicircle checkbox functionality."""
    viewer = make_napari_viewer()
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


def test_on_image_layer_changed_prevents_recursion(
    make_napari_viewer,
):
    """Test that the method prevents recursive calls using the guard flag."""

    viewer = make_napari_viewer()
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
    make_napari_viewer,
):
    """Test behavior when combobox has empty layer name."""

    viewer = make_napari_viewer()
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
    make_napari_viewer,
):
    """Test that harmonic spinbox maximum is set based on layer data."""
    viewer = make_napari_viewer()
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
    make_napari_viewer,
):
    """Test that the phasor data is accessible after layer change."""
    viewer = make_napari_viewer()
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
        with contextlib.suppress(Exception):
            plotter.on_image_layer_changed()
        plotter.deleteLater()

    # Verify guard flag is cleaned up even after exception
    assert (
        not hasattr(plotter, '_in_on_image_layer_changed')
        or not plotter._in_on_image_layer_changed
    )


def test_on_image_layer_changed_multiple_calls(
    make_napari_viewer,
):
    """Test multiple sequential calls to ensure plot is called each time."""

    viewer = make_napari_viewer()
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
        assert mock_draw.call_count >= 1

        # Reset mocks
        mock_hide_all.reset_mock()
        mock_show_tab.reset_mock()
        mock_draw.reset_mock()

        # Test tab change to FRET tab (index 6)
        fret_tab_index = 6
        plotter._on_tab_changed(fret_tab_index)

        mock_hide_all.assert_called_once()
        mock_show_tab.assert_called_once_with(plotter.fret_tab)
        assert mock_draw.call_count >= 1


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
    plotter = PlotterWidget(viewer)

    # Verify signal wiring by observing side effects from _on_tab_changed.
    with patch.object(plotter, '_hide_all_tab_artists') as mock_hide_all:
        plotter.tab_widget.setCurrentIndex(2)  # Change to filter tab
        assert mock_hide_all.call_count >= 1

    plotter.deleteLater()


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
    except AttributeError as err:
        raise AssertionError(
            "_set_components_visibility should handle missing components_tab gracefully"
        ) from err

    # Restore components_tab
    plotter.components_tab = original_components_tab

    # Remove the fret_tab attribute to test the hasattr check
    original_fret_tab = plotter.fret_tab
    del plotter.fret_tab

    # Should not raise an error when fret_tab doesn't exist
    try:
        plotter._set_fret_visibility(True)
        plotter._set_fret_visibility(False)
    except AttributeError as err:
        raise AssertionError(
            "_set_fret_visibility should handle missing fret_tab gracefully"
        ) from err

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
        except Exception as e:  # noqa: BLE001
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
        except Exception as e:  # noqa: BLE001
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
            assert not call[0][0]
        for call in mock_fret_vis.call_args_list:
            assert not call[0][0]

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


def test_phasor_plotter_mask_layer_ui_initialization(make_napari_viewer):
    """Test mask layer UI components exist and are initialized correctly."""
    viewer = make_napari_viewer()
    plotter = PlotterWidget(viewer)

    # Test mask layer combobox and label exist
    assert hasattr(plotter, 'mask_layer_combobox')
    assert hasattr(plotter, 'mask_layer_label')
    assert isinstance(plotter.mask_layer_combobox, QComboBox)
    assert isinstance(plotter.mask_layer_label, QLabel)

    # Test initial state - should have "None" as default
    assert plotter.mask_layer_combobox.currentText() == "None"


def test_phasor_plotter_apply_mask_to_phasor_data(make_napari_viewer):
    """Test that applying a mask sets G and S values outside mask to NaN."""
    viewer = make_napari_viewer()
    plotter = PlotterWidget(viewer)

    # Add image layer with phasors
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)

    # Get original G and S shape
    G_original = intensity_image_layer.metadata["G"]

    # Create a mask - need to match the spatial dimensions of G and S
    if G_original.ndim == 3:
        # Multi-harmonic: shape is (n_harmonics, height, width)
        mask_shape = G_original.shape[1:]
    else:
        # Single harmonic: shape is (height, width)
        mask_shape = G_original.shape

    mask_data = np.zeros(mask_shape, dtype=int)
    mask_data[mask_shape[0] // 2 :, :] = 1  # Mask in bottom half
    labels_layer = viewer.add_labels(mask_data, name="test_mask")

    # Apply mask
    plotter._apply_mask_to_phasor_data(labels_layer, intensity_image_layer)

    # Check that mask was stored in metadata
    assert 'mask' in intensity_image_layer.metadata

    # Check that G and S values outside mask are now NaN
    current_g = intensity_image_layer.metadata["G"]
    current_s = intensity_image_layer.metadata["S"]
    assert np.isnan(current_g).sum() > 0
    assert np.isnan(current_s).sum() > 0


def test_phasor_plotter_restore_original_phasor_data(make_napari_viewer):
    """Test restoring original phasor data removes mask effects."""
    viewer = make_napari_viewer()
    plotter = PlotterWidget(viewer)

    # Add image layer with phasors
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)

    # Get original data
    original_g = intensity_image_layer.metadata["G"].copy()
    original_s = intensity_image_layer.metadata["S"].copy()

    # Get the shape for the mask
    if original_g.ndim == 3:
        mask_shape = original_g.shape[1:]
    else:
        mask_shape = original_g.shape

    # Apply mask to modify data
    mask_data = np.zeros(mask_shape, dtype=int)
    mask_data[mask_shape[0] // 2 :, :] = 1
    labels_layer = viewer.add_labels(mask_data, name="test_mask")
    plotter._apply_mask_to_phasor_data(labels_layer, intensity_image_layer)

    # Verify data was modified (some values are NaN)
    assert np.isnan(intensity_image_layer.metadata["G"]).sum() > 0

    # Restore original data
    plotter._restore_original_phasor_data(intensity_image_layer)

    # Verify data was restored (no NaN values in original)
    np.testing.assert_array_almost_equal(
        intensity_image_layer.metadata["G"], original_g
    )
    np.testing.assert_array_almost_equal(
        intensity_image_layer.metadata["S"], original_s
    )


def test_mask_layer_rename_updates_combobox_and_assignments(
    make_napari_viewer,
):
    """Regression test for mask-layer rename synchronization in the plotter."""
    viewer = make_napari_viewer()

    layer1 = create_image_layer_with_phasors()
    layer2 = create_image_layer_with_phasors()
    viewer.add_layer(layer1)
    viewer.add_layer(layer2)

    # Build a mask compatible with phasor spatial dimensions.
    g_data = layer1.metadata["G"]
    mask_shape = g_data.shape[1:] if g_data.ndim == 3 else g_data.shape
    mask = np.ones(mask_shape, dtype=int)
    labels_layer = viewer.add_labels(mask, name="mask_before_rename")

    plotter = PlotterWidget(viewer)

    # Single-layer mode: selecting a mask updates the combobox and assignments.
    plotter.image_layers_checkable_combobox.setCheckedItems([layer1.name])
    plotter.mask_layer_combobox.setCurrentText("mask_before_rename")
    assert plotter.mask_layer_combobox.currentText() == "mask_before_rename"
    assert plotter._mask_assignments.get(layer1.name) == "mask_before_rename"

    # Renaming the mask layer should propagate to UI and assignments.
    labels_layer.name = "mask_after_rename"

    combo_items = [
        plotter.mask_layer_combobox.itemText(i)
        for i in range(plotter.mask_layer_combobox.count())
    ]
    assert "mask_after_rename" in combo_items
    assert "mask_before_rename" not in combo_items
    assert plotter.mask_layer_combobox.currentText() == "mask_after_rename"
    assert plotter._mask_assignments.get(layer1.name) == "mask_after_rename"

    # Re-running UI sync must not reset selection to None.
    plotter._update_mask_ui_mode()
    assert plotter.mask_layer_combobox.currentText() == "mask_after_rename"


def test_mask_ui_switches_to_button_when_multiple_layers_selected(
    make_napari_viewer,
):
    """Test that selecting multiple layers switches mask UI from combobox to button."""
    viewer = make_napari_viewer()
    layer1 = create_image_layer_with_phasors()
    layer2 = create_image_layer_with_phasors()
    viewer.add_layer(layer1)
    viewer.add_layer(layer2)
    plotter = PlotterWidget(viewer)

    # With only one layer selected (default primary), combobox should be visible.
    # Use not isHidden() because the plotter is not rendered in a window during tests,
    # so isVisible() would be False for all widgets regardless of their setVisible() state.
    plotter.image_layers_checkable_combobox.setCheckedItems([layer1.name])
    assert not plotter.mask_layer_combobox.isHidden()
    assert not plotter.mask_layer_label.isHidden()
    assert plotter.mask_assign_button.isHidden()

    # Select both layers - should switch to button mode
    plotter.image_layers_checkable_combobox.setCheckedItems(
        [layer1.name, layer2.name]
    )
    assert plotter.mask_layer_combobox.isHidden()
    assert plotter.mask_layer_label.isHidden()
    assert not plotter.mask_assign_button.isHidden()

    # Back to single selection - should revert to combobox mode
    plotter.image_layers_checkable_combobox.setCheckedItems([layer1.name])
    assert not plotter.mask_layer_combobox.isHidden()
    assert plotter.mask_assign_button.isHidden()


def test_mask_assignment_dialog_get_assignments(make_napari_viewer):
    """Test that MaskAssignmentDialog returns correct per-layer assignments."""
    from napari_phasors.plotter import MaskAssignmentDialog

    image_names = ["layer_A", "layer_B", "layer_C"]
    mask_names = ["mask_1", "mask_2"]
    current = {"layer_A": "mask_1", "layer_C": "mask_2"}

    dialog = MaskAssignmentDialog(
        image_layer_names=image_names,
        mask_layer_names=mask_names,
        current_assignments=current,
        parent=None,
    )

    # Initial assignments should reflect current_assignments
    assignments = dialog.get_assignments()
    assert assignments["layer_A"] == "mask_1"
    assert assignments["layer_B"] == "None"
    assert assignments["layer_C"] == "mask_2"


def test_mask_assignment_dialog_apply_all(make_napari_viewer):
    """Test that 'Set all to' applies the same mask to every layer in the dialog."""
    from napari_phasors.plotter import MaskAssignmentDialog

    image_names = ["layer_A", "layer_B"]
    mask_names = ["mask_1", "mask_2"]

    dialog = MaskAssignmentDialog(
        image_layer_names=image_names,
        mask_layer_names=mask_names,
        parent=None,
    )

    # Trigger apply-all by changing the combo
    dialog._apply_all_combo.setCurrentText("mask_2")

    assignments = dialog.get_assignments()
    assert assignments["layer_A"] == "mask_2"
    assert assignments["layer_B"] == "mask_2"


def test_apply_mask_assignments_different_masks_per_layer(make_napari_viewer):
    """Test that _apply_mask_assignments applies distinct masks to each layer."""
    viewer = make_napari_viewer()
    plotter = PlotterWidget(viewer)

    layer1 = create_image_layer_with_phasors()
    layer2 = create_image_layer_with_phasors()
    viewer.add_layer(layer1)
    viewer.add_layer(layer2)

    # Select both layers
    plotter.image_layers_checkable_combobox.setCheckedItems(
        [layer1.name, layer2.name]
    )

    # Build matching mask shapes from each layer's G data
    def make_mask(layer, fill_row_half):
        G = layer.metadata["G"]
        shape = G.shape[1:] if G.ndim == 3 else G.shape
        mask = np.zeros(shape, dtype=int)
        if fill_row_half:
            mask[shape[0] // 2 :, :] = 1  # bottom half
        else:
            mask[: shape[0] // 2, :] = 1  # top half
        return mask

    mask_data_1 = make_mask(layer1, fill_row_half=True)
    mask_data_2 = make_mask(layer2, fill_row_half=False)

    labels_layer1 = viewer.add_labels(mask_data_1, name="mask_bottom")
    labels_layer2 = viewer.add_labels(mask_data_2, name="mask_top")

    # Apply different masks to each layer
    assignments = {
        layer1.name: labels_layer1.name,
        layer2.name: labels_layer2.name,
    }
    plotter._apply_mask_assignments(assignments)

    # Both layers should now have masks stored
    assert 'mask' in layer1.metadata
    assert 'mask' in layer2.metadata

    # layer1 mask covers bottom half, so top half pixels should be NaN
    g1 = layer1.metadata["G"]
    if g1.ndim == 3:
        g1 = g1[0]
    assert np.isnan(g1[: g1.shape[0] // 2, :]).all()  # top half is NaN
    assert not np.isnan(g1[g1.shape[0] // 2 :, :]).all()  # bottom half kept

    # layer2 mask covers top half, so bottom half pixels should be NaN
    g2 = layer2.metadata["G"]
    if g2.ndim == 3:
        g2 = g2[0]
    assert np.isnan(g2[g2.shape[0] // 2 :, :]).all()  # bottom half is NaN
    assert not np.isnan(g2[: g2.shape[0] // 2, :]).all()  # top half kept

    # _mask_assignments should only contain non-None entries
    assert plotter._mask_assignments[layer1.name] == labels_layer1.name
    assert plotter._mask_assignments[layer2.name] == labels_layer2.name


def test_apply_mask_assignments_none_removes_mask(make_napari_viewer):
    """Test that assigning 'None' removes an existing mask from a layer."""
    viewer = make_napari_viewer()
    plotter = PlotterWidget(viewer)

    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)
    plotter.image_layers_checkable_combobox.setCheckedItems([layer.name])

    G = layer.metadata["G"]
    shape = G.shape[1:] if G.ndim == 3 else G.shape
    mask_data = np.ones(shape, dtype=int)
    labels_layer = viewer.add_labels(mask_data, name="full_mask")

    # First apply a mask
    plotter._apply_mask_to_phasor_data(labels_layer, layer)
    assert 'mask' in layer.metadata

    # Now assign None to remove it
    plotter._apply_mask_assignments({layer.name: "None"})

    assert 'mask' not in layer.metadata


def test_get_mask_for_layer_multi_mode(make_napari_viewer):
    """Test get_mask_for_layer returns per-layer assignment when multiple layers selected."""
    viewer = make_napari_viewer()
    layer1 = create_image_layer_with_phasors()
    layer2 = create_image_layer_with_phasors()
    viewer.add_layer(layer1)
    viewer.add_layer(layer2)
    plotter = PlotterWidget(viewer)

    G = layer1.metadata["G"]
    shape = G.shape[1:] if G.ndim == 3 else G.shape
    mask_data = np.ones(shape, dtype=int)
    labels_layer = viewer.add_labels(mask_data, name="my_mask")

    # Select both layers so we're in multi mode
    plotter.image_layers_checkable_combobox.setCheckedItems(
        [layer1.name, layer2.name]
    )

    # Assign mask only to layer1
    plotter._mask_assignments = {layer1.name: labels_layer.name}

    assert plotter.get_mask_for_layer(layer1.name) == labels_layer.name
    assert plotter.get_mask_for_layer(layer2.name) == "None"


def test_mask_assign_button_text_updates_with_count(make_napari_viewer):
    """Test that the assign button text shows how many layers have masks assigned."""
    viewer = make_napari_viewer()
    layer1 = create_image_layer_with_phasors()
    layer2 = create_image_layer_with_phasors()
    viewer.add_layer(layer1)
    viewer.add_layer(layer2)
    plotter = PlotterWidget(viewer)

    plotter.image_layers_checkable_combobox.setCheckedItems(
        [layer1.name, layer2.name]
    )

    # No masks assigned yet
    plotter._mask_assignments = {}
    plotter._update_mask_assign_button_text()
    assert plotter.mask_assign_button.text() == "Assign Masks..."

    # One mask assigned
    G = layer1.metadata["G"]
    shape = G.shape[1:] if G.ndim == 3 else G.shape
    labels_layer = viewer.add_labels(np.ones(shape, dtype=int), name="m")
    plotter._mask_assignments = {layer1.name: labels_layer.name}
    plotter._update_mask_assign_button_text()
    assert "1/2" in plotter.mask_assign_button.text()

    # Both masks assigned
    plotter._mask_assignments = {
        layer1.name: labels_layer.name,
        layer2.name: labels_layer.name,
    }
    plotter._update_mask_assign_button_text()
    assert "2/2" in plotter.mask_assign_button.text()


def test_toolbar_visibility_based_on_selection_mode(make_napari_viewer):
    """Test that toolbar visibility is controlled by selection mode."""
    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    plotter = PlotterWidget(viewer)

    # Initially in circular cursor mode, toolbar should be hidden
    assert plotter.selection_tab.selection_mode_combobox.currentIndex() == 0
    assert not plotter.selection_tab.is_manual_selection_mode()

    # In circular cursor mode, _show_tab_artists should not show toolbar
    with patch.object(plotter, '_set_selection_visibility') as mock_vis:
        plotter._show_tab_artists(plotter.selection_tab)
        # Should call _set_selection_visibility(False) when in circular cursor mode
        mock_vis.assert_called_once_with(False)

    # Switch to manual selection mode
    plotter.selection_tab.selection_mode_combobox.setCurrentText(
        "Manual Selection"
    )
    assert plotter.selection_tab.is_manual_selection_mode()

    # Now _show_tab_artists should show toolbar
    with patch.object(plotter, '_set_selection_visibility') as mock_vis:
        plotter._show_tab_artists(plotter.selection_tab)
        mock_vis.assert_called_once_with(True)

    # Mock _set_selection_visibility to track calls
    with patch.object(plotter, '_set_selection_visibility') as mock_vis:
        # Switch to circular cursor mode
        plotter.selection_tab.selection_mode_combobox.setCurrentText(
            "Circular Cursor"
        )

        # Should call _set_selection_visibility(False) to hide toolbar
        mock_vis.assert_called_once_with(False)


def test_is_manual_selection_mode_method(make_napari_viewer):
    """Test the is_manual_selection_mode() method."""
    viewer = make_napari_viewer()
    plotter = PlotterWidget(viewer)
    selection_widget = plotter.selection_tab

    # Initially in circular cursor mode (index 0)
    assert selection_widget.selection_mode_combobox.currentIndex() == 0
    assert not selection_widget.is_manual_selection_mode()

    # Switch to manual selection mode (index 2)
    selection_widget.selection_mode_combobox.setCurrentText("Manual Selection")
    assert selection_widget.is_manual_selection_mode()

    # Switch back to circular cursor mode
    selection_widget.selection_mode_combobox.setCurrentText("Circular Cursor")
    assert not selection_widget.is_manual_selection_mode()


def test_plot_colors_cleared_when_switching_from_manual_mode(
    make_napari_viewer,
):
    """Test that plot colors are cleared when switching from manual selection mode."""
    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    plotter = PlotterWidget(viewer)

    # Switch to manual selection mode
    plotter.selection_tab.selection_mode_combobox.setCurrentText(
        "Manual Selection"
    )

    # Make a manual selection (this would create colored points on plot)
    manual_selection = np.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 0])
    plotter.selection_tab.manual_selection_changed(manual_selection)

    # Mock plot method to verify it's called with selection_id_data=None
    with patch.object(plotter, 'plot') as mock_plot:
        # Switch to circular cursor mode
        plotter.selection_tab.selection_mode_combobox.setCurrentText(
            "Circular Cursor"
        )

        # plot should be called with selection_id_data=None to clear colors
        mock_plot.assert_called_once_with(selection_id_data=None)


def test_selection_tab_mode_switching_integration(make_napari_viewer):
    """Integration test for complete mode switching workflow."""
    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    plotter = PlotterWidget(viewer)
    selection_widget = plotter.selection_tab

    # Start in circular cursor mode
    assert selection_widget.selection_mode_combobox.currentIndex() == 0
    assert selection_widget.stacked_widget.currentIndex() == 0

    # Add a circular cursor
    circular_widget = selection_widget.circular_cursor_widget
    circular_widget._add_cursor()
    circular_widget._apply_selection()

    # Verify circular cursor layer exists (actual name is 'Cursor Selection:')
    circular_layer_name = f"Cursor Selection: {intensity_image_layer.name}"
    assert circular_layer_name in [layer.name for layer in viewer.layers]
    circular_layer = viewer.layers[circular_layer_name]
    assert circular_layer.visible is True

    # Switch to manual selection mode
    selection_widget.selection_mode_combobox.setCurrentText("Manual Selection")

    # Verify UI switched
    assert selection_widget.stacked_widget.currentIndex() == 4
    assert selection_widget.is_manual_selection_mode()

    # Verify circular cursor layer is now hidden
    assert circular_layer.visible is False

    # Make a manual selection
    manual_selection = np.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 0])
    selection_widget.manual_selection_changed(manual_selection)

    # Verify manual selection layer exists and is visible (no 'Selection ' prefix)
    manual_layer_name = f"MANUAL SELECTION #1: {intensity_image_layer.name}"
    assert manual_layer_name in [layer.name for layer in viewer.layers]
    manual_layer = viewer.layers[manual_layer_name]
    assert manual_layer.visible is True

    # Switch back to circular cursor mode
    selection_widget.selection_mode_combobox.setCurrentText("Circular Cursor")

    # Verify we're back in circular cursor mode
    assert not selection_widget.is_manual_selection_mode()

    # Verify visibility has switched back
    assert circular_layer.visible is True
    assert manual_layer.visible is False


def test_toolbar_hidden_when_switching_tabs(make_napari_viewer):
    """Test that toolbar is hidden when switching away from selection tab."""
    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    plotter = PlotterWidget(viewer)

    # Start at selection tab (index 3)
    plotter.tab_widget.setCurrentIndex(3)

    # Switch to manual selection mode to show toolbar
    plotter.selection_tab.selection_mode_combobox.setCurrentText(
        "Manual Selection"
    )

    # Mock _set_selection_visibility to track calls
    with patch.object(plotter, '_set_selection_visibility') as mock_vis:
        # Switch to a different tab (e.g., components tab at index 4)
        plotter.tab_widget.setCurrentIndex(4)

        # Should call _set_selection_visibility(False) to hide toolbar
        mock_vis.assert_called_with(False)


def test_toolbar_hidden_in_circular_cursor_mode(make_napari_viewer):
    """Test that toolbar is hidden when in circular cursor mode."""
    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    plotter = PlotterWidget(viewer)

    # Start at selection tab
    plotter.tab_widget.setCurrentIndex(3)

    # Start in circular cursor mode
    assert not plotter.selection_tab.is_manual_selection_mode()

    # Mock _set_selection_visibility to track calls
    with patch.object(plotter, '_set_selection_visibility') as mock_vis:
        # Trigger _show_tab_artists for selection tab
        plotter._show_tab_artists(plotter.selection_tab)

        # Should call _set_selection_visibility(False) because in circular cursor mode
        mock_vis.assert_called_once_with(False)


def test_toolbar_shown_only_in_manual_selection_mode(make_napari_viewer):
    """Test that toolbar is only shown in manual selection mode."""
    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    plotter = PlotterWidget(viewer)

    # Go to selection tab
    plotter.tab_widget.setCurrentIndex(3)

    # Test circular cursor mode - toolbar should be hidden
    plotter.selection_tab.selection_mode_combobox.setCurrentText(
        "Circular Cursor"
    )
    with patch.object(plotter, '_set_selection_visibility') as mock_vis:
        plotter._show_tab_artists(plotter.selection_tab)
        mock_vis.assert_called_once_with(False)

    # Test manual selection mode - toolbar should be shown
    plotter.selection_tab.selection_mode_combobox.setCurrentText(
        "Manual Selection"
    )
    with patch.object(plotter, '_set_selection_visibility') as mock_vis:
        plotter._show_tab_artists(plotter.selection_tab)
        mock_vis.assert_called_once_with(True)


def test_toolbar_visibility_on_mode_change_within_selection_tab(
    make_napari_viewer,
):
    """Test that toolbar visibility changes when switching modes within selection tab."""
    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    plotter = PlotterWidget(viewer)

    # Go to selection tab
    plotter.tab_widget.setCurrentIndex(3)

    # Start in circular cursor mode
    plotter.selection_tab.selection_mode_combobox.setCurrentText(
        "Circular Cursor"
    )

    # Mock _set_selection_visibility to track calls
    with patch.object(plotter, '_set_selection_visibility') as mock_vis:
        # Switch to manual selection mode
        plotter.selection_tab.selection_mode_combobox.setCurrentText(
            "Manual Selection"
        )

        # Should call _set_selection_visibility(True) to show toolbar
        assert mock_vis.call_count >= 1
        # Last call should be True
        assert mock_vis.call_args_list[-1][0][0]

    # Reset mock
    with patch.object(plotter, '_set_selection_visibility') as mock_vis:
        # Switch back to circular cursor mode
        plotter.selection_tab.selection_mode_combobox.setCurrentText(
            "Circular Cursor"
        )

        # Should call _set_selection_visibility(False) to hide toolbar
        assert mock_vis.call_count >= 1
        # Last call should be False
        assert not mock_vis.call_args_list[-1][0][0]


def test_circular_cursor_and_manual_selection_visibility_coordination(
    make_napari_viewer,
):
    """Test that circular cursor and manual selection visibility methods are properly coordinated."""
    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    plotter = PlotterWidget(viewer)

    # Go to selection tab
    plotter.tab_widget.setCurrentIndex(3)

    # Mock both visibility methods
    with (
        patch.object(plotter, '_set_selection_visibility') as mock_manual_vis,
        patch.object(
            plotter, '_set_selection_cursors_visibility'
        ) as mock_circular_vis,
    ):
        # Switch to circular cursor mode
        plotter.selection_tab.selection_mode_combobox.setCurrentText(
            "Circular Cursor"
        )
        plotter._show_tab_artists(plotter.selection_tab)

        # Manual selection toolbar should be hidden
        mock_manual_vis.assert_called_with(False)
        # Circular cursor should be shown
        mock_circular_vis.assert_called_with(True)

    # Reset mocks
    with (
        patch.object(plotter, '_set_selection_visibility') as mock_manual_vis,
    ):
        # Switch to manual selection mode
        plotter.selection_tab.selection_mode_combobox.setCurrentText(
            "Manual Selection"
        )
        plotter._show_tab_artists(plotter.selection_tab)

        # Manual selection toolbar should be shown
        mock_manual_vis.assert_called_with(True)
        # Circular cursor visibility method is only called in circular cursor mode
        # When in manual mode, circular cursors are hidden via layer visibility, not this method


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
        assert calls_comp[-1][0][0]
        # All fret calls should be False (hide fret)
        for call in calls_fret:
            assert not call[0][0]

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
            assert not call[0][0]
        # The last FRET call should be True (show FRET)
        assert calls_fret[-1][0][0]


def test_canvas_cleared_when_no_layer_selected(make_napari_viewer):
    """Test that canvas phasor data is cleared but semicircle/circle remains when no layer is selected."""
    viewer = make_napari_viewer()
    plotter = PlotterWidget(viewer)

    # Add a layer with phasors and verify plot is populated
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)

    # Verify layer is selected and data is plotted
    assert plotter.get_primary_layer_name() == intensity_image_layer.name
    assert plotter._g_array is not None
    assert plotter._s_array is not None
    assert plotter.colorbar is not None  # Should have colorbar for histogram

    # Store reference to semicircle/polar plot artists before clearing
    len(plotter.semi_circle_plot_artist_list)
    len(plotter.polar_plot_artist_list)

    # Deselect all layers (simulate no layer selected)
    plotter.image_layers_checkable_combobox.setCheckedItems([])

    # Trigger the layer change
    plotter.on_image_layer_changed()

    # Verify phasor data arrays are cleared
    assert plotter._g_array is None
    assert plotter._s_array is None
    assert plotter._g_original_array is None
    assert plotter._s_original_array is None
    assert plotter._harmonics_array is None

    # Verify colorbar is removed
    assert plotter.colorbar is None

    # Verify semicircle/polar plot remains (artists should still exist)
    if plotter.toggle_semi_circle:
        # Semicircle should have artists (lines, labels for lifetime ticks)
        assert len(plotter.semi_circle_plot_artist_list) > 0
    else:
        # Polar plot should have artists (lines and circles)
        assert len(plotter.polar_plot_artist_list) > 0

    # Verify axes are still configured properly
    assert plotter.canvas_widget.axes.get_xlabel() == "G"
    assert plotter.canvas_widget.axes.get_ylabel() == "S"
    assert plotter.canvas_widget.axes.get_aspect() == 1

    # Verify axes limits are reasonable (not cleared to default [0,1])
    xlim = plotter.canvas_widget.axes.get_xlim()
    ylim = plotter.canvas_widget.axes.get_ylim()
    if plotter.toggle_semi_circle:
        # Semicircle mode
        assert xlim[0] < 0 and xlim[1] > 1
        assert ylim[0] <= 0 and ylim[1] > 0.5
    else:
        # Full circle mode
        assert xlim[0] < -0.5 and xlim[1] > 0.5
        assert ylim[0] < -0.5 and ylim[1] > 0.5

    # Verify tab artists are hidden
    with (
        patch.object(plotter, '_set_components_visibility') as mock_comp_vis,
        patch.object(plotter, '_set_fret_visibility') as mock_fret_vis,
        patch.object(plotter, '_set_selection_visibility'),
    ):
        # This was already called during on_image_layer_changed, but verify the method works
        plotter._hide_all_tab_artists()

        # All should be called with False
        mock_comp_vis.assert_called_with(False)
        mock_fret_vis.assert_called_with(False)
        # Selection visibility is called in hide_all


def test_canvas_cleared_then_restored_with_new_layer(make_napari_viewer):
    """Test that canvas can be restored after being cleared."""
    viewer = make_napari_viewer()
    plotter = PlotterWidget(viewer)

    # Add first layer
    layer1 = create_image_layer_with_phasors()
    viewer.add_layer(layer1)

    # Verify data is loaded
    assert plotter._g_array is not None
    assert plotter._s_array is not None

    # Clear by deselecting
    plotter.image_layers_checkable_combobox.setCheckedItems([])
    plotter.on_image_layer_changed()

    # Verify cleared
    assert plotter._g_array is None
    assert plotter._s_array is None
    assert plotter.colorbar is None

    # Add second layer and select it
    layer2 = create_image_layer_with_phasors()
    viewer.add_layer(layer2)

    # Select the old layer
    plotter.image_layers_checkable_combobox.setCheckedItems([layer1.name])
    plotter.on_image_layer_changed()

    # Verify data is restored (new layer data)
    assert plotter._g_array is not None
    assert plotter._s_array is not None
    assert plotter.colorbar is not None
    assert plotter.get_primary_layer_name() == layer1.name


def test_artist_data_cleared_when_no_layer_selected(make_napari_viewer):
    """Test that biaplotter artists' internal data is properly cleared."""
    viewer = make_napari_viewer()
    plotter = PlotterWidget(viewer)

    # Add a layer to populate artist data
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)

    # Verify histogram artist has data
    histogram_artist = plotter.canvas_widget.artists['HISTOGRAM2D']
    scatter_artist = plotter.canvas_widget.artists['SCATTER']

    # Note: We can't directly check _data because the artists may not expose it,
    # but we can verify _remove_artists was called
    with (
        patch.object(histogram_artist, '_remove_artists') as mock_hist_remove,
        patch.object(scatter_artist, '_remove_artists') as mock_scatter_remove,
    ):
        # Deselect all layers
        plotter.image_layers_checkable_combobox.setCheckedItems([])
        plotter.on_image_layer_changed()

        # Verify _remove_artists was called on both artists
        mock_hist_remove.assert_called_once()
        mock_scatter_remove.assert_called_once()


def test_import_settings_filter_applies_to_all_selected_layers(
    make_napari_viewer,
):
    """Test that the import settings filter button applies to all selected layers."""
    viewer = make_napari_viewer()
    plotter = PlotterWidget(viewer)

    # Create three identical layers and add them to the viewer
    layer_a = create_image_layer_with_phasors()
    layer_b = create_image_layer_with_phasors()
    layer_c = create_image_layer_with_phasors()
    viewer.add_layer(layer_a)
    viewer.add_layer(layer_b)
    viewer.add_layer(layer_c)

    # Select all three layers
    plotter.image_layers_checkable_combobox.setCheckedItems(
        [layer_a.name, layer_b.name, layer_c.name]
    )

    # Build a minimal settings dict that represents an active filter
    imported_settings = {
        "filter": {"method": "median", "size": 3, "repeat": 1},
        "threshold": 1.0,
        "threshold_upper": None,
        "threshold_method": "Manual",
    }

    # Apply imported settings (the buggy code only touched the primary layer)
    plotter._apply_imported_settings(
        imported_settings, selected_tabs=["filter_tab"]
    )

    # All three layers must have the filter settings in their metadata
    for layer in (layer_a, layer_b, layer_c):
        settings = layer.metadata.get("settings", {})
        assert (
            "filter" in settings
        ), f"{layer.name}: 'filter' key missing from settings after import"
        assert (
            settings["filter"].get("method") == "median"
        ), f"{layer.name}: filter method not saved after import"
        assert (
            settings.get("threshold") == 1.0
        ), f"{layer.name}: threshold not saved after import"

    plotter.deleteLater()


def test_apply_calibration_if_needed_applies_to_all_selected_layers(
    make_napari_viewer,
):
    """Test that the apply calibration if needed button applies to all selected layers."""
    viewer = make_napari_viewer()
    plotter = PlotterWidget(viewer)

    layer_a = create_image_layer_with_phasors()
    layer_b = create_image_layer_with_phasors()
    viewer.add_layer(layer_a)
    viewer.add_layer(layer_b)

    # Select both layers
    plotter.image_layers_checkable_combobox.setCheckedItems(
        [layer_a.name, layer_b.name]
    )

    # Inject fake calibration parameters into both layers so the helper
    # believes they need calibration applied.
    for layer in (layer_a, layer_b):
        layer.metadata.setdefault("settings", {}).update(
            {
                "calibrated": True,
                "calibration_phase": [0.0],
                "calibration_modulation": [1.0],
            }
        )
        # calibration_applied is intentionally absent / False
        layer.metadata.pop("calibration_applied", None)

    # Patch the actual phasor transformation so the test stays fast & pure
    transformed_layers = []

    def _fake_transform(layer_name, phi_zero, mod_zero):
        transformed_layers.append(layer_name)
        # Mark as applied so the guard inside the loop works correctly
        viewer.layers[layer_name].metadata["calibration_applied"] = True

    with patch.object(
        plotter.calibration_tab,
        "_apply_phasor_transformation",
        side_effect=_fake_transform,
    ):
        plotter._apply_calibration_if_needed()

    # Both layers must have been transformed — not just the primary one
    assert layer_a.name in transformed_layers, (
        "layer_a was not calibrated "
        "(only primary layer was affected — bug regression)"
    )
    assert layer_b.name in transformed_layers, (
        "layer_b was not calibrated "
        "(only primary layer was affected — bug regression)"
    )

    plotter.deleteLater()


def test_copy_metadata_from_layer_applies_to_all_selected_layers(
    make_napari_viewer,
):
    """Test that the copy metadata from layer button applies to all selected layers."""
    viewer = make_napari_viewer()
    plotter = PlotterWidget(viewer)

    # Source layer whose settings we want to copy
    source_layer = create_image_layer_with_phasors()
    # Target layers — all are selected in the combobox
    layer_b = create_image_layer_with_phasors()
    layer_c = create_image_layer_with_phasors()

    for lyr in (source_layer, layer_b, layer_c):
        viewer.add_layer(lyr)

    # Pre-configure the source layer with specific settings we can assert on
    source_layer.metadata.setdefault("settings", {}).update(
        {
            "filter": {"method": "median", "size": 5, "repeat": 2},
            "threshold": 10.0,
            "threshold_upper": None,
            "threshold_method": "Manual",
        }
    )

    # Select only the target layers (source is NOT selected)
    plotter.image_layers_checkable_combobox.setCheckedItems(
        [layer_b.name, layer_c.name]
    )

    # Import filter settings from the source layer
    plotter._copy_metadata_from_layer(
        source_layer.name, selected_tabs=["filter_tab"]
    )

    # Both target layers must now carry the filter settings
    for layer in (layer_b, layer_c):
        settings = layer.metadata.get("settings", {})
        assert "filter" in settings, (
            f"{layer.name}: 'filter' key missing — "
            "settings were not copied to this layer"
        )
        assert settings["filter"].get("size") == 5, (
            f"{layer.name}: filter size not copied "
            "(only primary layer was affected — bug regression)"
        )
        assert (
            settings.get("threshold") == 10.0
        ), f"{layer.name}: threshold not copied from source layer"

    # Source layer's own settings must remain unchanged
    assert source_layer.metadata["settings"]["filter"]["size"] == 5

    plotter.deleteLater()


def test_phasor_center_settings_dialog_ui_states(qtbot):
    """Verify dialog correctly hides/shows mode selection based on layer count."""
    # single layer - mode should be hidden
    dialog1 = PhasorCenterLayerSettingsDialog(
        layer_labels=["One Layer"], display_mode="Merged"
    )
    qtbot.addWidget(dialog1)

    # Dialog title for single layer is simplified
    assert dialog1.windowTitle() == "Phasor Center Settings"
    assert dialog1._mode_container.isHidden()
    assert dialog1._merged_color_label.text() == "Center color:"

    # Multiple layers - mode should be visible
    dialog2 = PhasorCenterLayerSettingsDialog(
        layer_labels=["L1", "L2"], display_mode="Merged"
    )
    qtbot.addWidget(dialog2)
    assert dialog2.windowTitle() == "Phasor Center Settings (Multi-Layer)"
    assert not dialog2._mode_container.isHidden()
    assert dialog2._merged_color_label.text() == "Merged center color:"


def test_phasor_center_integration_and_dock_logic(make_napari_viewer, qtbot):
    """Test full integration of phasor centers: artist creation, dock switching, and table naming."""
    viewer = make_napari_viewer()
    plotter = PlotterWidget(viewer)
    qtbot.addWidget(plotter)

    # Add a layer with data
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)

    # Ensure dock exists
    plotter._add_analysis_dock_widget()

    # 1. Enable centers
    with patch.object(plotter._statistics_dock, 'raise_') as mock_raise:
        plotter.plotter_inputs_widget.phasor_center_checkbox.setChecked(True)

        # Verify artists created
        assert len(plotter._phasor_center_artists) > 0

        # Verify statistics page switch
        # Switch to Plot Settings tab, should show phasor center stats
        plotter.tab_widget.setCurrentWidget(plotter.settings_tab)
        assert (
            plotter._statistics_stack.currentIndex()
            == plotter._phasor_center_stats_page_idx
        )

        # Verify raise_ was called specifically on the statistics dock
        # because toggling centers ON raises it to visibility
        mock_raise.assert_called()

    # 2. Verify table content for single layer (should NOT say 'Merged', should be the layer name)
    table = plotter._phasor_center_stats_widget._layer_table
    assert table.rowCount() == 1
    assert table.item(0, 0).text() == layer.name

    # 3. Verify metadata update
    assert (
        layer.metadata.get('settings', {}).get('phasor_center_enabled') is True
    )

    # 4. Disable centers
    plotter.plotter_inputs_widget.phasor_center_checkbox.setChecked(False)
    assert len(plotter._phasor_center_artists) == 0
    assert table.rowCount() == 0
    assert layer.metadata['settings']['phasor_center_enabled'] is False

    plotter.deleteLater()


def _table_rows_by_name(table):
    """Return table rows as ``{name: (g, s, phase, mod)}`` with float values."""
    rows = {}
    for row in range(table.rowCount()):
        name = table.item(row, 0).text()
        rows[name] = (
            float(table.item(row, 1).text()),
            float(table.item(row, 2).text()),
            float(table.item(row, 3).text()),
            float(table.item(row, 4).text()),
        )
    return rows


def _set_layer_harmonic0_samples(layer, g_values, s_values, intensity_values):
    """Overwrite harmonic-1 samples for deterministic center test cases."""
    g_array = np.array(layer.metadata["G"], dtype=float, copy=True)
    s_array = np.array(layer.metadata["S"], dtype=float, copy=True)
    g_values = np.asarray(g_values, dtype=float)
    s_values = np.asarray(s_values, dtype=float)
    intensity_values = np.asarray(intensity_values, dtype=float)

    if g_array.ndim > intensity_values.ndim:
        g_array[0] = g_values
        s_array[0] = s_values
    else:
        g_array = g_values
        s_array = s_values

    layer.metadata["G"] = g_array
    layer.metadata["S"] = s_array
    layer.data = intensity_values


def test_phasor_center_multi_layer_merged_stats_name(make_napari_viewer):
    """Multi-layer merged mode should produce a single 'Merged' row."""
    viewer = make_napari_viewer()
    plotter = PlotterWidget(viewer)

    layer1 = create_image_layer_with_phasors()
    layer2 = create_image_layer_with_phasors()
    viewer.add_layer(layer1)
    viewer.add_layer(layer2)

    plotter.image_layers_checkable_combobox.setCheckedItems(
        [layer1.name, layer2.name]
    )
    plotter._process_layer_selection_change()

    plotter._phasor_center_display_mode = "Merged"
    plotter.plotter_inputs_widget.phasor_center_checkbox.setChecked(True)
    plotter._update_phasor_centers()

    layer_table = plotter._phasor_center_stats_widget._layer_table
    group_table = plotter._phasor_center_stats_widget._group_table

    assert len(plotter._phasor_center_artists) == 1
    assert layer_table.rowCount() == 1
    assert layer_table.item(0, 0).text() == "Merged"
    assert group_table.rowCount() == 0

    plotter.deleteLater()


def test_phasor_center_multi_layer_individual_stats(make_napari_viewer):
    """Individual mode should create one center/stat row per selected layer."""
    viewer = make_napari_viewer()
    plotter = PlotterWidget(viewer)

    layer1 = create_image_layer_with_phasors()
    layer2 = create_image_layer_with_phasors()
    viewer.add_layer(layer1)
    viewer.add_layer(layer2)

    plotter.image_layers_checkable_combobox.setCheckedItems(
        [layer1.name, layer2.name]
    )
    plotter._process_layer_selection_change()

    plotter._phasor_center_display_mode = "Individual layers"
    plotter.plotter_inputs_widget.phasor_center_checkbox.setChecked(True)
    plotter._update_phasor_centers()

    layer_table = plotter._phasor_center_stats_widget._layer_table
    group_table = plotter._phasor_center_stats_widget._group_table
    rows = _table_rows_by_name(layer_table)

    assert len(plotter._phasor_center_artists) == 2
    assert set(rows) == {layer1.name, layer2.name}
    assert group_table.rowCount() == 0

    c1 = plotter._compute_single_center(layer1)
    c2 = plotter._compute_single_center(layer2)
    assert c1 is not None
    assert c2 is not None
    np.testing.assert_allclose(rows[layer1.name][:2], c1, atol=1e-6)
    np.testing.assert_allclose(rows[layer2.name][:2], c2, atol=1e-6)

    plotter.deleteLater()


def test_phasor_center_grouped_median_uses_pooled_samples(make_napari_viewer):
    """Grouped mode should use pooled samples with selected median method."""
    viewer = make_napari_viewer()
    plotter = PlotterWidget(viewer)

    layer1 = create_image_layer_with_phasors(harmonic=[1])
    layer2 = create_image_layer_with_phasors(harmonic=[1])
    viewer.add_layer(layer1)
    viewer.add_layer(layer2)

    shape = layer1.data.shape
    g1 = np.zeros(shape, dtype=float)
    s1 = np.zeros(shape, dtype=float)
    m1 = np.ones(shape, dtype=float)

    g2 = np.full(shape, np.nan, dtype=float)
    s2 = np.full(shape, np.nan, dtype=float)
    m2 = np.ones(shape, dtype=float)
    g2[0, 0] = 1.0
    s2[0, 0] = 1.0
    g2[0, 1] = 1.0
    s2[0, 1] = 1.0

    _set_layer_harmonic0_samples(layer1, g1, s1, m1)
    _set_layer_harmonic0_samples(layer2, g2, s2, m2)

    plotter.image_layers_checkable_combobox.setCheckedItems(
        [layer1.name, layer2.name]
    )
    plotter._process_layer_selection_change()

    plotter._phasor_center_method = "median"
    plotter._phasor_center_display_mode = "Grouped"
    plotter._phasor_center_group_assignments = {
        layer1.name: 1,
        layer2.name: 1,
    }
    plotter._phasor_center_group_names = {1: "All layers"}

    plotter.plotter_inputs_widget.phasor_center_checkbox.setChecked(True)
    plotter._update_phasor_centers()

    layer_rows = _table_rows_by_name(
        plotter._phasor_center_stats_widget._layer_table
    )
    group_rows = _table_rows_by_name(
        plotter._phasor_center_stats_widget._group_table
    )

    assert len(plotter._phasor_center_artists) == 1
    assert set(layer_rows) == {layer1.name, layer2.name}
    assert set(group_rows) == {"All layers"}

    s_layer1 = plotter._get_layer_phasor_samples(layer1)
    s_layer2 = plotter._get_layer_phasor_samples(layer2)
    assert s_layer1 is not None
    assert s_layer2 is not None
    pooled_mean = np.concatenate([s_layer1[0], s_layer2[0]])
    pooled_g = np.concatenate([s_layer1[1], s_layer2[1]])
    pooled_s = np.concatenate([s_layer1[2], s_layer2[2]])

    pooled_center = plotter._compute_center_from_samples(
        pooled_mean,
        pooled_g,
        pooled_s,
    )
    assert pooled_center is not None
    np.testing.assert_allclose(
        group_rows["All layers"][:2], pooled_center, atol=1e-6
    )

    c1 = plotter._compute_single_center(layer1)
    c2 = plotter._compute_single_center(layer2)
    assert c1 is not None
    assert c2 is not None
    arithmetic_mean = (
        0.5 * (c1[0] + c2[0]),
        0.5 * (c1[1] + c2[1]),
    )
    assert not np.allclose(pooled_center, arithmetic_mean, atol=1e-6)

    plotter.deleteLater()
