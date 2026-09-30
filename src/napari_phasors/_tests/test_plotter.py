import contextlib
import warnings
from unittest.mock import patch

import numpy as np
import pytest
from napari.layers import Image
from qtpy.QtCore import QEvent, Qt
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
from napari_phasors._utils import WARNING_ICON_SIZE
from napari_phasors.calibration_tab import CalibrationWidget
from napari_phasors.components_tab import ComponentsWidget
from napari_phasors.filter_tab import FilterWidget
from napari_phasors.fret_tab import FretWidget
from napari_phasors.plotter import (
    CanvasWidget,
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


def test_phasor_plotter_without_layers(make_viewer_model, qtbot):
    """A plotter built on an empty viewer has every tab and control at its
    default, lays the settings out compactly, debounces canvas resizes, and
    changes settings without replotting while there is nothing to plot."""
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QLabel, QScrollArea

    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)
    qtbot.addWidget(plotter)

    # Basic widget structure tests
    assert plotter.viewer == viewer
    assert isinstance(plotter.layout(), QVBoxLayout)

    # Canvas widget tests
    assert isinstance(plotter.canvas_widget, CanvasWidget)
    assert plotter.canvas_widget.minimumSize().width() >= 300
    assert plotter.canvas_widget.minimumSize().height() >= 300
    assert plotter.canvas_widget.class_spinbox.value == 1

    # UI components tests (the layer combobox is a wrapper, not a QComboBox)
    assert hasattr(plotter, 'image_layer_with_phasor_features_combobox')
    assert isinstance(plotter.harmonic_spinbox, QSpinBox)
    assert plotter.harmonic_spinbox.minimum() == 1
    assert plotter.harmonic_spinbox.value() == 1

    # Import buttons tests
    assert isinstance(plotter.import_from_layer_button, QPushButton)
    assert plotter.import_from_layer_button.text() == "Layer"
    assert isinstance(plotter.import_from_file_button, QPushButton)
    assert plotter.import_from_file_button.text() == "OME-TIFF File"

    # Tab widget tests
    assert isinstance(plotter.tab_widget, QTabWidget)
    tab_names = [
        plotter.tab_widget.tabText(i)
        for i in range(plotter.tab_widget.count())
    ]
    assert tab_names == [
        "Plot Settings",
        "Calibration",
        "Filter",
        "Selection",
        "Components",
        "Phasor Mapping",
        "FRET",
    ]
    assert hasattr(plotter, 'settings_tab')
    assert hasattr(plotter, 'phasor_mapping_tab')
    assert isinstance(plotter.filter_tab, FilterWidget)
    assert isinstance(plotter.selection_tab, SelectionWidget)
    assert isinstance(plotter.calibration_tab, CalibrationWidget)
    assert isinstance(plotter.components_tab, ComponentsWidget)
    assert isinstance(plotter.fret_tab, FretWidget)

    # Test settings tab inputs widget
    for name in (
        'plot_type_combobox',
        'colormap_combobox',
        'number_of_bins_spinbox',
        'semi_circle_checkbox',
        'white_background_checkbox',
        'log_scale_checkbox',
        'marker_size_spinbox',
        'marker_transparency_spinbox',
        'marker_color_button',
    ):
        assert hasattr(plotter.plotter_inputs_widget, name)

    # Test default property values
    assert plotter.harmonic == 1
    assert plotter.plot_type == 'HISTOGRAM2D'
    assert plotter.histogram_colormap == 'jet'
    assert plotter.toggle_semi_circle  # Should default to semi-circle mode
    assert plotter.white_background  # Should default to white background

    plot_type_combobox = plotter.plotter_inputs_widget.plot_type_combobox
    assert [
        plot_type_combobox.itemText(i)
        for i in range(plot_type_combobox.count())
    ] == [
        "Density Plot (2D Histogram)",
        "Dot Plot (Scatter)",
        "Contour Plot",
        "None",
    ]
    # The plot-type combobox must not force the Plot Settings tab wide just
    # to display its longest item text ("Density Plot (2D Histogram)") --
    # it should truncate/elide instead of setting a large minimum width.
    assert plot_type_combobox.minimumSizeHint().width() < 200
    assert plotter.plotter_inputs_widget.colormap_combobox.count() > 0

    # Test canvas widget artists and initial plot elements
    assert 'SCATTER' in plotter.canvas_widget.artists
    assert 'HISTOGRAM2D' in plotter.canvas_widget.artists
    assert plotter.colorbar is None
    assert plotter.minimumSize().width() >= 300
    assert plotter.minimumSize().height() >= 300
    assert plotter.canvas_widget.axes.get_aspect() == 1
    # Initial axes limits (semi-circle mode)
    xlim = plotter.canvas_widget.axes.get_xlim()
    ylim = plotter.canvas_widget.axes.get_ylim()
    assert xlim[0] == -0.1 and xlim[1] == 1.1
    assert ylim[0] == -0.1 and ylim[1] == 0.7

    # Long, non-wrapping labels used to force the Plot Settings tab (and
    # its inner scroll area) far wider than necessary. "Full Polar Plot
    # (Spectral Phasor)" must wrap instead of setting a ~240px-wide floor.
    assert plotter.plotter_inputs_widget.label_5.wordWrap()
    # "Load and Apply Settings from:" sits inline with the Layer/OME-TIFF
    # buttons, so it is intentionally not word-wrapped; the scroll policy
    # keeps that row from forcing the tab permanently wide.
    import_labels = [
        label
        for label in plotter.settings_tab.findChildren(QLabel)
        if label.text() == "Load and Apply Settings from:"
    ]
    assert len(import_labels) == 1
    scroll_area = plotter.settings_tab.findChild(QScrollArea)
    assert scroll_area is not None
    assert scroll_area.horizontalScrollBarPolicy() == Qt.ScrollBarAsNeeded

    # Resize/Show events on the watched containers (re)start the debounced
    # resize timer rather than firing the resize immediately.
    # ``_resize_canvas_timer`` is parented to the widget precisely so it
    # cannot outlive it (see the PySide6 teardown segfault notes).
    assert not plotter._resize_canvas_timer.isActive()
    handled = plotter.eventFilter(
        plotter.canvas_container, QEvent(QEvent.Resize)
    )
    assert plotter._resize_canvas_timer.isActive()
    # eventFilter should not consume the event for the container itself.
    assert handled is False
    # An event on an unrelated object must not start the timer.
    plotter._resize_canvas_timer.stop()
    plotter.eventFilter(plotter, QEvent(QEvent.Resize))
    assert not plotter._resize_canvas_timer.isActive()

    # Harmonic-bound refresh safely ignores unavailable selections.
    layer = create_image_layer_with_phasors()
    layer.metadata.pop("harmonics", None)
    plotter._update_harmonic_bounds([])
    plotter._update_harmonic_bounds([layer])

    # Switching plot types hides and shows the matching inputs.
    piw = plotter.plotter_inputs_widget

    def assert_inputs_shown(colormap, bins, log_scale, markers, contour=None):
        assert plotter._colormap_row_widget.isHidden() is not colormap
        assert piw.number_of_bins_spinbox.isHidden() is not bins
        assert piw.log_scale_checkbox.isHidden() is not log_scale
        for marker_input in (
            piw.marker_size_spinbox,
            piw.marker_transparency_spinbox,
            piw.marker_color_button,
        ):
            assert marker_input.isHidden() is not markers
        if contour is not None:
            assert piw.contour_levels_spinbox.isHidden() is not contour
            assert piw.contour_linewidth_spinbox.isHidden() is not contour

    assert plotter.plot_type == 'HISTOGRAM2D'
    assert_inputs_shown(
        colormap=True, bins=True, log_scale=True, markers=False
    )
    plot_type_combobox.setCurrentText("Dot Plot (Scatter)")
    assert_inputs_shown(
        colormap=False, bins=False, log_scale=False, markers=True
    )
    plot_type_combobox.setCurrentText("Contour Plot")
    assert_inputs_shown(
        colormap=True, bins=True, log_scale=False, markers=False, contour=True
    )
    plot_type_combobox.setCurrentText("None")
    assert_inputs_shown(
        colormap=False,
        bins=False,
        log_scale=False,
        markers=False,
        contour=False,
    )
    plot_type_combobox.setCurrentText("Density Plot (2D Histogram)")

    # Without a layer, modifying values does not replot. Patched after
    # construction to avoid Qt teardown issues from constructing
    # PlotterWidget inside a patch context manager.
    with patch.object(plotter, '_set_active_artist_and_plot') as mock_plot:
        plotter.plot_type = 'SCATTER'
        plotter.histogram_colormap = 'viridis'
        plotter.toggle_semi_circle = False
        plotter.white_background = False
        piw.semi_circle_checkbox.setChecked(True)
        piw.white_background_checkbox.setChecked(False)
        mock_plot.assert_not_called()

    # Test property setters
    plotter.harmonic = 3
    assert plotter.harmonic == 3
    assert plotter.harmonic_spinbox.value() == 3
    plotter.plot_type = 'SCATTER'
    assert plotter.plot_type == 'SCATTER'
    plotter.histogram_colormap = 'viridis'
    assert plotter.histogram_colormap == 'viridis'
    plotter.white_background = True
    assert plotter.white_background
    assert piw.white_background_checkbox.isChecked()
    plotter.toggle_semi_circle = False
    assert not plotter.toggle_semi_circle
    assert piw.semi_circle_checkbox.isChecked()

    # A contour selection is remembered even with nothing drawn.
    plotter.plot_type = 'CONTOUR'
    plotter._on_selector_applied(np.array([1, 0, 1]))
    assert plotter._last_contour_color_indices is not None

    # _clear_contour_plot falls back to the tracked collections when the
    # CONTOUR artist is missing.
    contour_artist = plotter.canvas_widget.artists.pop('CONTOUR')
    try:

        class MockSubCol:
            def __init__(self):
                self.removed = False

            def remove(self):
                self.removed = True

        class MockCollection:
            def __init__(self):
                self.removed = False
                self.collections = [MockSubCol()]

            def remove(self):
                self.removed = True

        plotter._contour_collections = [MockCollection()]
        # A dummy "contour_plot_element" artist and a legend on the axes.
        plotter.canvas_widget.axes.plot(
            [0, 1], [0, 1], label="contour_plot_element"
        )
        plotter.canvas_widget.axes.legend()
        plotter._clear_contour_plot()
        assert len(plotter._contour_collections) == 0

        # _update_contour_plot returns early when the artist is missing.
        assert plotter._update_contour_plot([0.5], [0.5]) is None
    finally:
        plotter.canvas_widget.artists['CONTOUR'] = contour_artist

    plotter.deleteLater()


def test_phasor_plotter_initialization_with_layer(make_viewer_model):
    """A layer present when the plotter is built is selected, plotted as a
    density plot with its harmonics and default settings, and replotted by
    every layer change, guarded against re-entry and exceptions."""
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

    # A density plot with its colorbar, labelled axes and semicircle ticks.
    assert plotter.plot_type == 'HISTOGRAM2D'
    assert plotter.canvas_widget.artists['HISTOGRAM2D'] is not None
    assert plotter.colorbar is not None
    assert plotter.canvas_widget.axes.get_xlabel() == "G"
    assert plotter.canvas_widget.axes.get_ylabel() == "S"
    if plotter.toggle_semi_circle:
        assert len(plotter.semi_circle_plot_artist_list) > 0

    # Settings are initialized in the layer metadata with their defaults.
    settings = intensity_image_layer.metadata['settings']
    for key in (
        'harmonic',
        'semi_circle',
        'white_background',
        'plot_type',
        'colormap',
        'number_of_bins',
        'log_scale',
        'marker_size',
        'marker_alpha',
        'marker_color',
    ):
        assert key in settings
    assert settings['harmonic'] == 1
    assert settings['semi_circle']
    assert settings['white_background']
    assert settings['plot_type'] == 'HISTOGRAM2D'
    assert settings['colormap'] == 'jet'
    assert plotter.plotter_inputs_widget.marker_size_spinbox.value() == 50
    assert settings['number_of_bins'] == 150
    assert not settings['log_scale']
    assert settings['marker_size'] == 50
    assert settings['marker_alpha'] == 0.5
    assert settings['marker_color'] == '#1f77b4'

    # Each layer change keeps the harmonic maximum, the selected layer and
    # its phasor data, and replots once.
    plotter.on_image_layer_changed()
    assert plotter.harmonic_spinbox.maximum() == expected_max_harmonic
    layer_name = (
        plotter.image_layer_with_phasor_features_combobox.currentText()
    )
    assert layer_name == intensity_image_layer.name
    assert "G" in viewer.layers[layer_name].metadata
    assert "S" in viewer.layers[layer_name].metadata
    with patch.object(plotter, 'plot') as mock_plot:
        plotter.on_image_layer_changed()
        plotter.on_image_layer_changed()
        plotter.on_image_layer_changed()
        assert mock_plot.call_count == 3

    # The guard flag is cleaned up even if plotting raises...
    with patch.object(
        plotter, 'plot', side_effect=Exception("Test exception")
    ):
        with contextlib.suppress(Exception):
            plotter.on_image_layer_changed()
    assert (
        not hasattr(plotter, '_in_on_image_layer_changed')
        or not plotter._in_on_image_layer_changed
    )

    # ...and while it is set, a re-entrant call returns early, leaving it.
    plotter._in_on_image_layer_changed = True
    with patch.object(plotter, 'plot') as mock_plot:
        plotter.on_image_layer_changed()
        mock_plot.assert_not_called()
    assert plotter._in_on_image_layer_changed
    plotter._in_on_image_layer_changed = False

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

    # A reset requested while one is already running is ignored.
    plotter._resetting_layer_choices = True
    plotter.image_layers_checkable_combobox.clear()
    plotter.reset_layer_choices()
    assert plotter.image_layers_checkable_combobox.count() == 0
    plotter._resetting_layer_choices = False

    plotter.deleteLater()


def test_switch_plot_type_after_removing_scatter_layer(make_viewer_model):
    """Regression: SCATTER → remove layer → HISTOGRAM2D → add layer.

    Removing the last layer calls ``artist._remove_artists()`` which empties
    biaplotter's ``_mpl_artists`` but used to leave ``_color_indices``
    populated. When a new layer was later added and the active artist was
    switched, biaplotter's ``_colorize`` indexed into the now-empty
    ``_mpl_artists`` and raised ``KeyError: 'scatter'``.
    """
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)

    # Draw a scatter plot so the Scatter artist gets real color indices.
    layer_1 = create_image_layer_with_phasors()
    viewer.add_layer(layer_1)
    plotter.plot_type = 'SCATTER'
    scatter_artist = plotter.canvas_widget.artists['SCATTER']
    assert scatter_artist._color_indices is not None

    # Remove the layer — this empties the mpl artists.
    viewer.layers.remove(layer_1)
    assert scatter_artist._mpl_artists == {}
    # Invariant restored: no mpl artists => no stale color indices.
    assert scatter_artist._color_indices is None

    # Switch to density and import a new layer. Previously this raised
    # KeyError: 'scatter' from biaplotter's _colorize.
    plotter.plot_type = 'HISTOGRAM2D'
    layer_2 = create_image_layer_with_phasors()
    viewer.add_layer(layer_2)

    assert plotter.plot_type == 'HISTOGRAM2D'
    assert plotter.canvas_widget.active_artist.__class__.__name__ == (
        'Histogram2D'
    )

    plotter.deleteLater()


def test_switching_between_two_layers(make_viewer_model, qtbot):
    """Plot settings carry over when switching layers and are restored when
    switching back, and unrelated layer events replace neither the primary
    layer nor an explicitly empty selection."""
    viewer = make_viewer_model()
    layer_a = create_image_layer_with_phasors()
    layer_a.name = "source_a"
    layer_b = create_image_layer_with_phasors()
    layer_b.name = "source_b"
    viewer.add_layer(layer_a)
    viewer.add_layer(layer_b)
    plotter = PlotterWidget(viewer)
    primary = plotter.image_layer_with_phasor_features_combobox
    combo = plotter.image_layers_checkable_combobox

    # Checking a second layer keeps the primary, so its data is not
    # reloaded (the auto-selected primary used to be announced again).
    assert plotter.get_primary_layer_name() == "source_a"
    with patch.object(
        plotter, '_apply_layer_data', wraps=plotter._apply_layer_data
    ) as apply_layer_data:
        combo.model().item(1).setCheckState(Qt.Checked)
    apply_layer_data.assert_not_called()
    assert combo.checkedItems() == ["source_a", "source_b"]
    combo.setCheckedItems(["source_a"])

    primary.setCurrentText(layer_a.name)
    plotter.plot_type = 'SCATTER'
    plotter.histogram_colormap = 'viridis'
    plotter.toggle_semi_circle = False
    plotter.white_background = False

    # Switching to the other layer keeps the current settings...
    primary.setCurrentText(layer_b.name)
    assert plotter.plot_type == 'SCATTER'
    assert plotter.histogram_colormap == 'viridis'
    assert not plotter.toggle_semi_circle
    assert not plotter.white_background

    # ...and switching back restores the first layer's.
    primary.setCurrentText(layer_a.name)
    assert plotter.plot_type == 'SCATTER'
    assert plotter.histogram_colormap == 'viridis'
    assert not plotter.toggle_semi_circle
    assert not plotter.white_background

    combo.setCheckedItems(["source_a", "source_b"])
    combo.setPrimaryLayer("source_b")
    derived = viewer.add_image(np.zeros((2, 2)), name="derived")
    assert combo.checkedItems() == ["source_a", "source_b"]
    assert combo.getPrimaryLayer() == "source_b"

    combo.setCheckedItems([])
    viewer.layers.remove(derived)
    assert combo.checkedItems() == []
    assert combo.getPrimaryLayer() == ""

    plotter.deleteLater()


def test_plot_settings_controls(make_viewer_model):
    """The semicircle, background, colormap and log scale controls redraw
    the canvas, and every setting change is written to the layer's
    metadata."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)
    plotter = PlotterWidget(viewer)
    piw = plotter.plotter_inputs_widget

    # Initially semicircle should be enabled (default), with y-limits
    # starting from 0 or slightly below.
    assert plotter.toggle_semi_circle
    assert not piw.semi_circle_checkbox.isChecked()
    initial_ylim = plotter.canvas_widget.axes.get_ylim()
    assert initial_ylim[0] <= 0.1
    # The full circle extends well below y=0, with polar plot lines.
    piw.semi_circle_checkbox.setChecked(True)
    plotter.toggle_semi_circle = False
    new_ylim = plotter.canvas_widget.axes.get_ylim()
    assert new_ylim[0] < initial_ylim[0]
    assert abs(new_ylim[0]) > 0.2
    assert len(plotter.canvas_widget.axes.get_lines()) > 0
    # Toggle back to semicircle
    piw.semi_circle_checkbox.setChecked(False)
    plotter.toggle_semi_circle = True
    assert plotter.canvas_widget.axes.get_ylim()[0] >= new_ylim[0]

    # The background turns non-white (or transparent) and back.
    white_bg_checkbox = piw.white_background_checkbox
    initial_bg_state = white_bg_checkbox.isChecked()
    initial_bg_color = plotter.canvas_widget.axes.get_facecolor()
    new_bg_state = not initial_bg_state
    white_bg_checkbox.setChecked(new_bg_state)
    plotter.white_background = new_bg_state
    assert plotter.white_background == new_bg_state
    assert white_bg_checkbox.isChecked() == new_bg_state
    new_bg_color = plotter.canvas_widget.axes.get_facecolor()
    is_white = all(component > 0.9 for component in new_bg_color[:3])
    if new_bg_state:
        assert is_white
    else:
        assert not is_white or new_bg_color[3] < 0.1
    white_bg_checkbox.setChecked(initial_bg_state)
    plotter.white_background = initial_bg_state
    np.testing.assert_allclose(
        plotter.canvas_widget.axes.get_facecolor(), initial_bg_color, atol=0.1
    )

    # Choosing another colormap in the combobox applies it.
    colormap_combobox = piw.colormap_combobox
    assert plotter.histogram_colormap == 'jet'
    assert colormap_combobox.count() > 1
    test_colormap = next(
        colormap_combobox.itemText(i)
        for i in range(colormap_combobox.count())
        if colormap_combobox.itemText(i) not in {'jet', 'Select color...'}
    )
    colormap_combobox.setCurrentText(test_colormap)
    assert plotter.histogram_colormap == test_colormap
    assert colormap_combobox.currentText() == test_colormap

    # The colorbar survives colormap and log scale changes.
    assert plotter.colorbar is not None
    new_index = (colormap_combobox.currentIndex() + 1) % (
        colormap_combobox.count()
    )
    colormap_combobox.setCurrentIndex(new_index)
    plotter.histogram_colormap = colormap_combobox.currentText()
    assert plotter.colorbar is not None
    log_scale_checkbox = piw.log_scale_checkbox
    initial_log_state = log_scale_checkbox.isChecked()
    log_scale_checkbox.setChecked(not initial_log_state)
    assert log_scale_checkbox.isChecked() != initial_log_state
    assert plotter.colorbar is not None
    plotter.histogram_log_scale = not initial_log_state
    assert plotter.histogram_log_scale == (not initial_log_state)
    log_scale_checkbox.setChecked(initial_log_state)
    assert log_scale_checkbox.isChecked() == initial_log_state

    # Log normalization and non-positive ylim warnings are suppressed.
    plotter.plot_type = 'HISTOGRAM2D'
    plotter.histogram_log_scale = True
    with warnings.catch_warnings(record=True) as caught_warnings:
        plotter.plot()
        for w in caught_warnings:
            msg = str(w.message)
            if (
                "Log normalization applied" in msg
                or "non-positive ylim" in msg
            ):
                pytest.fail(f"Warning was not suppressed: {msg}")

    # Every control writes its setting to the layer metadata.
    plotter.harmonic = 3
    assert layer.metadata['settings']['harmonic'] == 3
    piw.plot_type_combobox.setCurrentText("Dot Plot (Scatter)")
    assert layer.metadata['settings']['plot_type'] == 'SCATTER'
    piw.colormap_combobox.setCurrentText('viridis')
    assert layer.metadata['settings']['colormap'] == 'viridis'
    piw.number_of_bins_spinbox.setValue(200)
    assert layer.metadata['settings']['number_of_bins'] == 200
    piw.log_scale_checkbox.setChecked(False)
    piw.log_scale_checkbox.setChecked(True)
    assert layer.metadata['settings']['log_scale']
    piw.semi_circle_checkbox.setChecked(True)
    assert not layer.metadata['settings']['semi_circle']
    piw.white_background_checkbox.setChecked(False)
    assert not layer.metadata['settings']['white_background']
    piw.marker_size_spinbox.setValue(30)
    assert layer.metadata['settings']['marker_size'] == 30
    # Marker transparency is stored as its complement, alpha.
    piw.marker_transparency_spinbox.setValue(0.8)
    assert layer.metadata['settings']['marker_alpha'] == pytest.approx(0.2)
    plotter._marker_color = '#ff0000'
    plotter._update_setting_in_metadata('marker_color', '#ff0000')
    assert layer.metadata['settings']['marker_color'] == '#ff0000'

    # So do the properties.
    plotter.harmonic = 2
    plotter.plot_type = 'HISTOGRAM2D'
    plotter.histogram_colormap = 'jet'
    plotter.toggle_semi_circle = True
    plotter.white_background = True
    plotter.plot_type = 'SCATTER'
    plotter.histogram_colormap = 'viridis'
    plotter.toggle_semi_circle = False
    plotter.white_background = False
    assert layer.metadata['settings']['harmonic'] == 2
    assert layer.metadata['settings']['plot_type'] == 'SCATTER'
    assert layer.metadata['settings']['colormap'] == 'viridis'
    assert not layer.metadata['settings']['semi_circle']
    assert not layer.metadata['settings']['white_background']

    plotter.deleteLater()


def test_layers_added_to_an_open_plotter(make_viewer_model):
    """Only layers with phasor features are offered; the first one added is
    selected and plotted with default settings, and removing it empties
    the plotter without replotting."""
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)
    combobox = plotter.image_layers_checkable_combobox

    # Initially no layers with phasor features
    assert combobox.count() == 0

    # A regular image layer neither replots nor updates the combobox.
    with (
        patch.object(plotter, '_set_active_artist_and_plot') as mock_plot,
        patch.object(plotter, 'on_image_layer_changed') as mock_layer_changed,
    ):
        viewer.add_layer(Image(np.random.random((10, 10))))
        mock_plot.assert_not_called()
        mock_layer_changed.assert_not_called()
        assert combobox.count() == 0
        assert (
            plotter.image_layer_with_phasor_features_combobox.currentText()
            == ''
        )

    # With no layer name there is nothing to plot.
    with patch.object(plotter, 'plot') as mock_plot:
        plotter.on_image_layer_changed()
        mock_plot.assert_not_called()

    # The first source added to an empty viewer is selected and plotted,
    # and a layer without settings metadata gets the defaults.
    layer = create_image_layer_with_phasors()
    layer.name = "first_source"
    if 'settings' in layer.metadata:
        del layer.metadata['settings']
    with patch.object(plotter, 'plot') as mock_plot:
        viewer.add_layer(layer)
        mock_plot.assert_called_once()
    assert plotter.get_selected_layer_names() == ["first_source"]
    assert plotter.get_primary_layer_name() == "first_source"
    assert combobox.count() == 1
    assert (
        plotter.image_layer_with_phasor_features_combobox.currentText()
        == "first_source"
    )
    assert 'settings' in layer.metadata
    assert layer.metadata['settings']['harmonic'] == 1
    assert layer.metadata['settings']['semi_circle']
    assert layer.metadata['settings']['white_background']
    assert layer.metadata['settings']['plot_type'] == 'HISTOGRAM2D'

    # Regression: unchecking the auto-selected layer, as a click does, left
    # its plot on screen because the combobox never announced it.
    item = combobox.model().item(0)
    item.setCheckState(Qt.Unchecked)
    assert plotter.get_primary_layer_name() == ""
    assert plotter._g_array is None
    assert plotter.colorbar is None
    # Re-applying the empty selection does not tear the plot down again.
    with patch.object(plotter, '_clear_all_tab_artists') as teardown:
        plotter.on_image_layer_changed()
    teardown.assert_not_called()
    item.setCheckState(Qt.Checked)
    assert plotter.get_primary_layer_name() == "first_source"
    assert plotter._g_array is not None

    # Combobox should still have only the layer with phasor features
    viewer.add_layer(Image(np.random.random((10, 10))))
    assert combobox.count() == 1

    # Removing the phasor layer empties the combobox without replotting.
    with patch.object(plotter, 'plot') as mock_plot:
        viewer.layers.remove(layer)
        mock_plot.assert_not_called()
    assert combobox.count() == 0

    # The scatter control is worded as transparency for consistency with
    # the rest of the plugin, while ``marker_alpha`` keeps matplotlib's
    # opacity meaning so settings saved by older versions still restore.
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)
    plotter.image_layer_with_phasor_features_combobox.setCurrentText(
        layer.name
    )
    plotter.on_image_layer_changed()
    piw = plotter.plotter_inputs_widget
    assert piw.label_marker_transparency.text() == "Transparency:"
    assert piw.marker_transparency_spinbox.minimum() == 0.0
    piw.marker_transparency_spinbox.setValue(0.25)
    assert layer.metadata['settings']['marker_alpha'] == pytest.approx(0.75)
    # Restoring an alpha from metadata puts its complement in the control.
    layer.metadata['settings']['marker_alpha'] = 0.4
    plotter._restore_plot_settings_from_metadata()
    assert piw.marker_transparency_spinbox.value() == pytest.approx(0.6)
    piw.plot_type_combobox.setCurrentText("Dot Plot (Scatter)")
    assert plotter.canvas_widget.artists['SCATTER'].alpha == pytest.approx(0.4)

    plotter.deleteLater()


def test_contour_plot_of_several_layers(make_viewer_model, monkeypatch):
    """Contour mode draws one layer or several, cleans up when left, and
    draws one contour per group (skipping unassigned layers, with a single
    warning) or per layer in its own style."""
    from napari.utils import notifications

    viewer = make_viewer_model()
    layer1 = create_image_layer_with_phasors()
    layer2 = create_image_layer_with_phasors()
    viewer.add_layer(layer1)
    viewer.add_layer(layer2)
    plotter = PlotterWidget(viewer)
    combobox = plotter.image_layers_checkable_combobox

    def contour_artists_in_axes():
        return [
            artist
            for artist in plotter.canvas_widget.axes.collections
            if artist.get_label() == 'contour_plot_element'
        ]

    # Single-layer contour render with real phasor data.
    combobox.setCheckedItems([layer1.name])
    plotter._process_layer_selection_change()
    plotter.plot_type = 'CONTOUR'
    plotter.plot()
    assert len(plotter._contour_collections) > 0
    assert len(contour_artists_in_axes()) > 0

    # Multi-layer contour render should also succeed without exceptions.
    combobox.setCheckedItems([layer1.name, layer2.name])
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

    # Grouped mode creates one contour collection per group.
    plotter._contour_display_mode = "Grouped"
    plotter._contour_group_assignments = {layer1.name: 1, layer2.name: 1}
    plotter._contour_group_names = {1: "Group 1"}
    plotter.plot()
    assert len(plotter._contour_collections) == 1
    assert len(contour_artists_in_axes()) > 0

    # A layer in no group draws no contour instead of joining group 1.
    warnings_seen = []
    monkeypatch.setattr(
        notifications, "show_warning", lambda msg: warnings_seen.append(msg)
    )
    plotter._contour_group_assignments = {layer1.name: 1}
    plotter.plot()
    assert len(plotter._contour_collections) == 1
    assert len(warnings_seen) == 1
    assert layer2.name in warnings_seen[0]
    # Redrawing the same selection does not repeat the warning.
    plotter.plot()
    assert len(warnings_seen) == 1
    # Assigning the missing layer clears the warned state.
    plotter._contour_group_assignments[layer2.name] = 1
    plotter.plot()
    assert len(warnings_seen) == 1
    assert plotter._warned_unassigned.get("contour plot") is None

    # Individual layers, in colormap and solid styles, with a legend.
    plotter._contour_show_legend = True
    plotter._contour_display_mode = "Individual layers"
    plotter._contour_layer_styles = {
        layer1.name: {"mode": "colormap", "colormap": "viridis"},
        layer2.name: {"mode": "solid", "color": (1.0, 0.0, 0.0)},
    }
    plotter.plot()
    contour_artist = plotter.canvas_widget.artists['CONTOUR']
    assert len(contour_artist._contour_collections) == 2
    assert plotter.canvas_widget.axes.get_legend() is not None
    # Solid style without an explicit color falls back to a default.
    plotter._contour_layer_styles = {
        layer1.name: {"mode": "solid"},
        layer2.name: {"mode": "solid"},
    }
    plotter.plot()
    assert len(contour_artist._contour_collections) == 2

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


def test_single_layer_selection_restores_its_full_harmonic_range(
    make_viewer_model, qtbot
):
    """Removing a restrictive peer restores the remaining layer's harmonics."""
    viewer = make_viewer_model()
    layer_a = create_image_layer_with_phasors(harmonic=[1, 2, 3])
    layer_a.name = "multi_harmonic"
    layer_b = create_image_layer_with_phasors(harmonic=[1])
    layer_b.name = "single_harmonic"
    viewer.add_layer(layer_a)
    viewer.add_layer(layer_b)
    plotter = PlotterWidget(viewer)
    combo = plotter.image_layers_checkable_combobox

    combo.setCheckedItems(["multi_harmonic", "single_harmonic"])
    plotter._layer_selection_timer.stop()
    plotter._process_layer_selection_change()
    assert plotter.harmonic_spinbox.maximum() == 1

    combo.setCheckedItems(["multi_harmonic"])
    plotter._layer_selection_timer.stop()
    plotter._process_layer_selection_change()
    assert plotter.harmonic_spinbox.minimum() == 1
    assert plotter.harmonic_spinbox.maximum() == 3


def test_performance_settings(make_viewer_model, qtbot):
    """The experimental Performance section switches the thread pools, sizes
    them against a memory budget, and sets the phasor storage precision,
    each with a hint describing the result."""
    from napari_phasors import _parallel, _utils

    previous_items = _parallel.parallel_items_enabled()
    previous_bands = _parallel.parallel_bands_enabled()
    previous_fraction = _parallel.memory_fraction()
    previous_budget_enabled = _parallel.memory_budget_enabled()
    previous_dtype = _utils.phasor_storage_dtype()
    # "native" -- keep whatever was read -- is the shipped default, and the
    # precision combobox takes its initial value from the setting, so pin it
    # before building the widget rather than assuming the ambient state.
    assert _utils.PHASOR_STORAGE_DTYPES[0] == "native"
    _utils.set_phasor_storage_dtype("native")
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)
    try:
        # ``error_label`` is the object name napari's stylesheet paints with
        # the warning triangle; borrowing it keeps the icon theme-correct.
        assert plotter.experimental_warning_icon.objectName() == "error_label"
        assert plotter.experimental_warning_label.text() == "Experimental"
        assert "report it" in plotter.experimental_warning_icon.toolTip()
        # The triangle is also rendered directly, since the stylesheet only
        # reaches the label where napari's theme is applied. Its *logical*
        # size must fit the 18 px box less 2 px of padding napari leaves for
        # ``#error_label``: a pre-scaled high-DPI pixmap would be clipped
        # into an unrecognisable wedge, as ``QIcon.pixmap`` already takes a
        # logical size and tags the result with its device pixel ratio.
        pixmap = plotter.experimental_warning_icon.pixmap()
        assert pixmap is not None and not pixmap.isNull()
        dpr = pixmap.devicePixelRatio() or 1.0
        assert (pixmap.width() / dpr) == pytest.approx(WARNING_ICON_SIZE)
        assert (pixmap.height() / dpr) == pytest.approx(WARNING_ICON_SIZE)
        assert "image: none" in plotter.experimental_warning_icon.styleSheet()

        # The two switches drive their own scope, independently.
        hint = plotter.parallel_processing_hint
        assert plotter.parallel_items_checkbox.isChecked() is False
        assert plotter.parallel_bands_checkbox.isChecked() is False
        assert "sequentially on one thread" in hint.text()

        plotter.parallel_items_checkbox.setChecked(True)
        plotter.parallel_bands_checkbox.setChecked(True)
        # Captured while both are on, so the comparison below survives a
        # single-core runner or a NAPARI_PHASORS_WORKERS override.
        bands_baseline = _parallel.default_workers(
            n_items=8, workers=8, scope=_parallel.BANDS
        )

        # Turning images off must leave band splitting alone, and vice versa.
        plotter.parallel_items_checkbox.setChecked(False)
        assert _parallel.parallel_items_enabled() is False
        assert _parallel.parallel_bands_enabled() is True
        assert (
            _parallel.default_workers(
                n_items=8, workers=8, scope=_parallel.ITEMS
            )
            == 1
        )
        assert (
            _parallel.default_workers(
                n_items=8, workers=8, scope=_parallel.BANDS
            )
            == bands_baseline
        )
        assert "one at a time" in hint.text()

        plotter.parallel_items_checkbox.setChecked(True)
        plotter.parallel_bands_checkbox.setChecked(False)
        assert _parallel.parallel_items_enabled() is True
        assert _parallel.parallel_bands_enabled() is False
        assert (
            _parallel.default_workers(
                n_items=8, workers=8, scope=_parallel.BANDS
            )
            == 1
        )
        assert "in one piece" in hint.text()

        # Both off is the fully sequential plugin.
        plotter.parallel_items_checkbox.setChecked(False)
        assert "sequentially on one thread" in hint.text()

        # The budget spinbox is what every memory-sized pool is measured
        # against, once the budget is switched on.
        assert plotter.memory_budget_checkbox.isChecked() is False
        assert plotter.memory_budget_spinbox.isEnabled() is False
        assert plotter.memory_budget_spinbox.value() == round(
            _parallel.DEFAULT_MEMORY_FRACTION * 100
        )
        assert _parallel.items_for_memory(1 << 20) is None

        plotter.memory_budget_checkbox.setChecked(True)
        assert plotter.memory_budget_spinbox.isEnabled() is True
        assert _parallel.memory_budget_enabled() is True
        plotter.memory_budget_spinbox.setValue(10)
        assert _parallel.memory_fraction() == pytest.approx(0.10)
        tight = _parallel.items_for_memory(1 << 20)
        plotter.memory_budget_spinbox.setValue(80)
        assert _parallel.memory_fraction() == pytest.approx(0.80)
        roomy = _parallel.items_for_memory(1 << 20)
        # Only meaningful where free memory could be read at all.
        if tight is not None and roomy is not None:
            assert roomy > tight

        plotter.memory_budget_checkbox.setChecked(False)
        assert plotter.memory_budget_spinbox.isEnabled() is False
        assert _parallel.memory_budget_enabled() is False
        assert _parallel.items_for_memory(1 << 20) is None

        # The precision control drives what new layers store their arrays
        # as.
        combobox = plotter.phasor_precision_combobox
        assert combobox.currentData() == "native"
        assert _utils.cast_phasor_storage(np.ones((2, 2)))[0].dtype == (
            np.float64
        )
        combobox.setCurrentText("float32 (half memory)")
        assert _utils.phasor_storage_dtype() == "float32"
        assert _utils.cast_phasor_storage(np.ones((2, 2)))[0].dtype == (
            np.float32
        )
        assert "float32" in hint.text()
        combobox.setCurrentText("As read")
        assert _utils.phasor_storage_dtype() == "native"
    finally:
        _parallel.set_parallel_items_enabled(previous_items)
        _parallel.set_parallel_bands_enabled(previous_bands)
        _parallel.set_memory_fraction(previous_fraction)
        _parallel.set_memory_budget_enabled(previous_budget_enabled)
        _utils.set_phasor_storage_dtype(previous_dtype)


def test_phasor_precision_setting():
    """An unknown precision name is rejected, and downcasting never touches
    labels nor upcasts what is already smaller."""
    from napari_phasors import _utils

    previous = _utils.phasor_storage_dtype()
    try:
        with pytest.raises(ValueError, match="unknown phasor storage"):
            _utils.set_phasor_storage_dtype("float8")
        assert _utils.phasor_storage_dtype() == previous

        _utils.set_phasor_storage_dtype("float32")
        counts = np.ones((2, 2), dtype=np.uint16)
        already = np.ones((2, 2), dtype=np.float32)
        cast_counts, cast_already, nothing = _utils.cast_phasor_storage(
            counts, already, None
        )
        assert cast_counts is counts
        assert cast_already is already
        assert nothing is None
    finally:
        _utils.set_phasor_storage_dtype(previous)


def test_switch_from_scatter_to_histogram2d_with_manual_selection(
    make_viewer_model,
):
    """Test switching from Scatter to Histogram 2D after making a manual selection."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)
    plotter = PlotterWidget(viewer)

    plotter.plot_type = 'SCATTER'

    # Make a manual selection in scatter mode
    plotter.selection_tab.selection_mode_combobox.setCurrentText(
        'Manual Selection'
    )
    selector = plotter.canvas_widget.selectors['RECTANGLE']
    selector.create_selector()

    class Event:
        def __init__(self, x, y):
            self.xdata = x
            self.ydata = y

    selector.on_select(Event(0.1, 0.1), Event(0.9, 0.9))
    selector.apply_selection()

    assert plotter.selection_tab.selection_id is not None
    assert plotter.selection_tab.selection_id != "None"

    # Switching to Histogram 2D should not raise TypeError
    sel_id = plotter.selection_tab.selection_id
    plotter.plotter_inputs_widget.plot_type_combobox.setCurrentText(
        "Density Plot (2D Histogram)"
    )
    assert plotter.plot_type == 'HISTOGRAM2D'
    assert plotter.selection_tab.selection_id == sel_id
    # Selection in layer metadata is preserved
    assert (
        sel_id in layer.metadata["settings"]["selections"]["manual_selections"]
    )
    hist_artist = plotter.canvas_widget.artists['HISTOGRAM2D']
    assert hist_artist.color_indices is not None
    assert np.any(hist_artist.color_indices > 0)

    # Switching back to Scatter also preserves selection
    plotter.plotter_inputs_widget.plot_type_combobox.setCurrentText(
        "Dot Plot (Scatter)"
    )
    assert plotter.plot_type == 'SCATTER'
    assert plotter.selection_tab.selection_id == sel_id
    scatter_artist = plotter.canvas_widget.artists['SCATTER']
    assert scatter_artist.color_indices is not None
    assert np.any(scatter_artist.color_indices > 0)

    plotter.deleteLater()


def test_contour_plot_of_one_layer(make_viewer_model):
    """Contour mode draws the layer, recolours its axes, computes its own
    limits and bins, and draws a group of one; the active-artist helper
    handles NONE and unknown artist names."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)
    plotter = PlotterWidget(viewer)

    plotter.plot_type = 'CONTOUR'
    plotter.plot()

    # Color snapshot and color application in CONTOUR mode.
    saved_colors = plotter._capture_plot_colors()
    assert len(saved_colors["axes"]) >= 1
    plotter._apply_plot_colors("black")
    ax = plotter.canvas_widget.artists['CONTOUR'].ax
    assert ax.xaxis.label.get_color() == "black"

    # Default axes limits in CONTOUR mode with the semicircle off.
    plotter.toggle_semi_circle = False
    plotter._redefine_axes_limits()
    xlim = plotter.canvas_widget.axes.get_xlim()
    ylim = plotter.canvas_widget.axes.get_ylim()
    assert xlim[0] < xlim[1]
    assert ylim[0] < ylim[1]
    # SCATTER mode with HISTOGRAM2D data available
    plotter.plot_type = 'SCATTER'
    plotter._redefine_axes_limits()
    # Tall axes (aspect <= 1) still get contour bins.
    plotter.plot_type = 'CONTOUR'
    plotter.canvas_widget.axes.set_xlim(0, 1)
    plotter.canvas_widget.axes.set_ylim(-5, 5)  # aspect = 1 / 10 <= 1
    features = plotter.get_features()
    assert features is not None
    x, y = features
    plotter._update_contour_plot(x, y)
    contour_artist = plotter.canvas_widget.artists['CONTOUR']
    assert contour_artist.bins is not None

    # Grouped mode skips an empty group, with a solid group colour...
    plotter._contour_display_mode = "Grouped"
    plotter._contour_group_assignments = {layer.name: 1}
    plotter._contour_group_names = {1: "Group 1", 2: "Empty Group 2"}
    plotter._contour_group_styles = {
        1: {"mode": "solid", "color": (0.0, 0.5, 1.0)},
        2: {"mode": "solid"},
    }
    plotter.plot()
    assert len(contour_artist._contour_collections) == 1
    # ...or a solid style without an explicit colour.
    plotter._contour_group_colors.clear()
    plotter._contour_group_styles = {1: {"mode": "solid"}}
    plotter.plot()
    assert len(contour_artist._contour_collections) == 1

    # _set_active_artist_and_plot follows a plot type that differs from
    # the combobox, and handles NONE and unregistered artist names.
    plotter.plot_type = 'HISTOGRAM2D'
    plotter._set_active_artist_and_plot('SCATTER', x, y)
    assert plotter.plot_type == 'SCATTER'
    plotter._set_active_artist_and_plot('NONE', x, y)
    assert plotter.canvas_widget.active_artist is None
    plotter._set_active_artist_and_plot('UNKNOWN_TYPE', x, y)
    assert plotter.canvas_widget.active_artist is None

    plotter.deleteLater()


def assert_run_row_is_pinned(tab, button, autoupdate=None):
    """Assert a tab's primary button sits under its scroll area, not in it.

    The analysis buttons are pinned below the scrolling settings so they stay
    reachable however far the settings above them have grown, with the
    "Autoupdate" switch on the same line to their right.
    """
    from qtpy.QtWidgets import QScrollArea

    scroll = tab.findChild(QScrollArea)
    assert scroll is not None

    ancestors = []
    node = button.parentWidget()
    while node is not None:
        ancestors.append(node)
        node = node.parentWidget()
    assert scroll.widget() not in ancestors

    row = button.parentWidget().layout()
    assert row.indexOf(button) >= 0
    if autoupdate is not None:
        assert row.indexOf(autoupdate) > row.indexOf(button)

    tab_layout = tab.layout()
    node = button
    while node is not None and tab_layout.indexOf(node) < 0:
        node = node.parentWidget()
    assert node is not None
    assert tab_layout.indexOf(node) > tab_layout.indexOf(scroll)


def test_closed_plotter_ignores_tab_changes(make_viewer_model, qtbot):
    """Closing disconnects the tab switch, which Qt emits while it deletes
    the tab pages, so the analysis redraws behind it cannot run then."""
    plotter = PlotterWidget(make_viewer_model())
    qtbot.addWidget(plotter)
    visibility_changes = []
    plotter.phasor_mapping_tab.on_tab_visibility_changed = (
        visibility_changes.append
    )
    tabs = plotter.tab_widget

    tabs.setCurrentIndex((tabs.currentIndex() + 1) % tabs.count())
    assert len(visibility_changes) == 1

    plotter.close()
    tabs.setCurrentIndex((tabs.currentIndex() + 1) % tabs.count())
    assert len(visibility_changes) == 1
