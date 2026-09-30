from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from napari_phasors._tests.test_plotter import (  # noqa: E501
    create_image_layer_with_phasors,
)
from napari_phasors._utils import analysis_layer_name, component_analysis_label
from napari_phasors.plotter import (
    PhasorCenterLayerSettingsDialog,
    PlotterWidget,
)


def _lp_name(component, source):
    return analysis_layer_name(component_analysis_label(component), source)


def test_plotter_without_a_layer(make_viewer_model, qtbot):
    """Layout, sizing, dock helpers and every no-layer path of a bare
    plotter."""
    from napari.layers import Labels
    from qtpy.QtWidgets import QSizePolicy

    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)
    qtbot.addWidget(plotter)

    # Tab pages drop the platform's wide default margins: stacked default
    # margins cost over 100 px of the dock's height, which made tabs scroll
    # while the dock still had room.
    margin = PlotterWidget._TAB_PAGE_MARGIN
    for index in range(plotter.tab_widget.count()):
        page = plotter.tab_widget.widget(index)
        assert page.layout().contentsMargins().top() == margin
        assert page.layout().contentsMargins().bottom() == margin
        assert page.layout().contentsMargins().left() == margin
    # The dock wrapper and the Plot Settings scroll-area container add none,
    # bar the thin strip that keeps the tab bar from being clipped on top.
    analysis_margins = plotter.analysis_widget.layout().contentsMargins()
    assert analysis_margins.top() == PlotterWidget._TAB_BAR_TOP_MARGIN
    assert analysis_margins.bottom() == 0
    assert analysis_margins.left() == 0
    assert plotter.plotter_inputs_widget.layout().contentsMargins().top() == 0

    # napari's forced ``Maximum`` vertical policy is set back to Expanding:
    # QtViewerDockWidget (napari >= 0.9) overwrites the vertical size policy
    # of every docked widget with Maximum, which has no grow flag.
    widgets = (
        plotter,
        plotter.analysis_widget,
        plotter.histogram_container,
        plotter.statistics_container,
    )
    for widget in widgets:
        widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
    plotter._restore_expanding_dock_policies()
    for widget in widgets:
        assert (
            widget.sizePolicy().verticalPolicy() == QSizePolicy.Expanding
        ), widget

    # The corner/split helpers no-op safely when there is no Qt main window:
    # no corners captured yet, and no hosting dock for the parent walk.
    plotter._restore_bottom_corners()
    assert not hasattr(plotter, '_original_bottom_corners')
    assert plotter._find_plotter_dock() is None
    plotter._split_analysis_below_plotter()

    # A bare widget has no bar over it, so nothing is reserved.
    assert plotter._title_bar_overlap() == 0
    plotter._reserve_title_bar_overlap()
    assert plotter.layout().contentsMargins().top() == 0

    # With nowhere to hand spare height, the widget stays free to grow.
    assert plotter._spare_height_can_be_reused() is False
    plotter.setMaximumHeight(500)
    plotter._update_useful_height_limit(plotter.canvas_container.width())
    assert plotter.maximumHeight() == PlotterWidget._NO_HEIGHT_LIMIT
    # Nothing to re-split, so the deferred claim is a no-op.
    plotter._claim_useful_height()

    # The canvas is sized from the aspect-locked axes plus fixed pixel
    # overheads, so the plot fills the container tightly.
    plotter.canvas_widget.resize(520, 470)
    plotter.canvas_widget.canvas.resize(500, 400)
    w, h = plotter._canvas_size_for_ratio(1.0, 800, 700)
    assert 0 < w <= 800
    assert 0 < h <= 700
    # The horizontal overhead (y-label + colorbar) exceeds the vertical one,
    # so a square plot needs a wider-than-tall canvas.
    assert w > h
    # Semicircle ratio also stays within the available box.
    w2, h2 = plotter._canvas_size_for_ratio(1.5, 800, 700)
    assert 0 < w2 <= 800 and 0 < h2 <= 700

    # Aspect-fit fallbacks: undrawn figure, no room for the axes, and a
    # canvas that cannot report numeric sizes.
    def fit(ratio, aw, ah):
        if aw / ah >= ratio:
            return ah * ratio, ah
        return aw, aw / ratio

    with patch.object(plotter.canvas_widget.canvas, 'width', return_value=0):
        assert plotter._canvas_size_for_ratio(1.0, 900, 300) == fit(
            1.0, 900, 300
        )
        assert plotter._canvas_size_for_ratio(1.5, 300, 900) == fit(
            1.5, 300, 900
        )
    plotter.canvas_widget.canvas.resize(500, 400)
    assert plotter._canvas_size_for_ratio(1.0, 60, 40) == fit(1.0, 60, 40)
    with patch.object(
        plotter.canvas_widget.canvas, 'width', return_value="bogus"
    ):
        assert plotter._canvas_size_for_ratio(1.0, 900, 300) == fit(
            1.0, 900, 300
        )

    # _resize_canvas_to_available_space fixes the canvas within its
    # container using the computed target size.
    plotter.canvas_container.resize(500, 400)
    plotter.canvas_widget.canvas.resize(480, 350)
    plotter._resize_canvas_to_available_space()
    assert 0 < plotter.canvas_widget.width() <= 500
    assert 0 < plotter.canvas_widget.height() <= 400

    # Pressing Home clears the stored user zoom so later replots use the
    # default limits, not the stale zoom.
    toolbar = getattr(plotter.canvas_widget, "toolbar", None)
    if toolbar is not None and getattr(toolbar, "_actions", {}).get("home"):
        plotter._user_axes_limits = ((0.2, 0.4), (0.2, 0.4))
        toolbar._actions["home"].trigger()
        assert plotter._user_axes_limits is None

    # No layer: nothing to refresh, mask, or intersect.
    plotter.refresh_phasor_data()
    assert plotter._g_array is None
    plotter._on_mask_layer_changed("None")
    assert plotter._get_common_harmonics([]) is None

    # _apply_layer_data with no layer and toggle_semi_circle=False uses the
    # polar plot.
    plotter.toggle_semi_circle = False
    plotter._apply_layer_data("", reset_zoom=True, sync_frequency=False)
    assert plotter._g_array is None

    # _on_mask_data_changed early-returns when no layers are selected.
    mask = Labels(np.ones((4, 4), dtype=np.uint8), name="mask")
    viewer.add_layer(mask)

    class _Event:
        source = mask

    plotter._on_mask_data_changed(_Event())


def test_canvas_cleared_and_restored_with_the_layer_selection(
    make_viewer_model,
):
    """Deselecting every layer clears the phasor data and artists but keeps
    the semicircle; selecting a layer again restores them."""
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)

    layer1 = create_image_layer_with_phasors()
    viewer.add_layer(layer1)

    # The layer is selected and its data is plotted.
    assert plotter.get_primary_layer_name() == layer1.name
    assert plotter._g_array is not None
    assert plotter._s_array is not None
    assert plotter.colorbar is not None  # Should have colorbar for histogram

    # Deselecting all layers clears the biaplotter artists' internal data.
    histogram_artist = plotter.canvas_widget.artists['HISTOGRAM2D']
    scatter_artist = plotter.canvas_widget.artists['SCATTER']
    with (
        patch.object(histogram_artist, '_remove_artists') as mock_hist_remove,
        patch.object(scatter_artist, '_remove_artists') as mock_scatter_remove,
    ):
        plotter.image_layers_checkable_combobox.setCheckedItems([])
        plotter.on_image_layer_changed()
        mock_hist_remove.assert_called_once()
        mock_scatter_remove.assert_called_once()

    # Phasor data arrays and the colorbar are cleared.
    assert plotter._g_array is None
    assert plotter._s_array is None
    assert plotter._g_original_array is None
    assert plotter._s_original_array is None
    assert plotter._harmonics_array is None
    assert plotter.colorbar is None

    # The semicircle/polar plot remains.
    if plotter.toggle_semi_circle:
        assert len(plotter.semi_circle_plot_artist_list) > 0
    else:
        assert len(plotter.polar_plot_artist_list) > 0

    # The axes are still configured, with limits not reset to [0, 1].
    assert plotter.canvas_widget.axes.get_xlabel() == "G"
    assert plotter.canvas_widget.axes.get_ylabel() == "S"
    assert plotter.canvas_widget.axes.get_aspect() == 1
    xlim = plotter.canvas_widget.axes.get_xlim()
    ylim = plotter.canvas_widget.axes.get_ylim()
    if plotter.toggle_semi_circle:
        assert xlim[0] < 0 and xlim[1] > 1
        assert ylim[0] <= 0 and ylim[1] > 0.5
    else:
        assert xlim[0] < -0.5 and xlim[1] > 0.5
        assert ylim[0] < -0.5 and ylim[1] > 0.5

    # Tab artists are hidden.
    with (
        patch.object(plotter, '_set_components_visibility') as mock_comp_vis,
        patch.object(plotter, '_set_fret_visibility') as mock_fret_vis,
        patch.object(plotter, '_set_selection_visibility'),
    ):
        plotter._hide_all_tab_artists()
        mock_comp_vis.assert_called_with(False)
        mock_fret_vis.assert_called_with(False)

    # Selecting a layer again restores the plot.
    layer2 = create_image_layer_with_phasors()
    viewer.add_layer(layer2)
    plotter.image_layers_checkable_combobox.setCheckedItems([layer1.name])
    plotter.on_image_layer_changed()
    assert plotter._g_array is not None
    assert plotter._s_array is not None
    assert plotter.colorbar is not None
    assert plotter.get_primary_layer_name() == layer1.name


def test_apply_layer_data_with_no_layer_clears_state(make_viewer_model):
    """_apply_layer_data with empty layer_name resets arrays and redraws."""
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)

    # Confirm there's data first
    assert plotter._g_array is not None

    # Now invoke with no layer name — this must hit the empty-path code.
    plotter._apply_layer_data("", reset_zoom=True, sync_frequency=True)

    assert plotter._g_array is None
    assert plotter._s_array is None
    assert plotter._g_original_array is None
    assert plotter._s_original_array is None
    assert plotter._harmonics_array is None

    plotter.deleteLater()


def test_settings_reach_every_selected_layer(make_viewer_model):
    """Imported settings, calibration and settings copied from another layer
    apply to every selected layer, not just the primary one."""
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)

    layer_a = create_image_layer_with_phasors()
    layer_b = create_image_layer_with_phasors()
    layer_c = create_image_layer_with_phasors()
    source_layer = create_image_layer_with_phasors()
    for lyr in (layer_a, layer_b, layer_c, source_layer):
        viewer.add_layer(lyr)

    # The import settings filter button applies to all selected layers.
    plotter.image_layers_checkable_combobox.setCheckedItems(
        [layer_a.name, layer_b.name, layer_c.name]
    )
    imported_settings = {
        "filter": {"method": "median", "size": 3, "repeat": 1},
        "threshold": 1.0,
        "threshold_upper": None,
        "threshold_method": "Manual",
    }
    plotter._apply_imported_settings(
        imported_settings, selected_tabs=["filter_tab"]
    )
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

    # Calibration is applied to all selected layers that need it.
    plotter.image_layers_checkable_combobox.setCheckedItems(
        [layer_a.name, layer_b.name]
    )
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
    assert layer_a.name in transformed_layers, (
        "layer_a was not calibrated "
        "(only primary layer was affected - bug regression)"
    )
    assert layer_b.name in transformed_layers, (
        "layer_b was not calibrated "
        "(only primary layer was affected - bug regression)"
    )

    # Copying settings from an unselected layer reaches every target.
    source_layer.metadata.setdefault("settings", {}).update(
        {
            "filter": {"method": "median", "size": 5, "repeat": 2},
            "threshold": 10.0,
            "threshold_upper": None,
            "threshold_method": "Manual",
        }
    )
    plotter.image_layers_checkable_combobox.setCheckedItems(
        [layer_b.name, layer_c.name]
    )
    plotter._copy_metadata_from_layer(
        source_layer.name, selected_tabs=["filter_tab"]
    )
    for layer in (layer_b, layer_c):
        settings = layer.metadata.get("settings", {})
        assert "filter" in settings, (
            f"{layer.name}: 'filter' key missing - "
            "settings were not copied to this layer"
        )
        assert settings["filter"].get("size") == 5, (
            f"{layer.name}: filter size not copied "
            "(only primary layer was affected - bug regression)"
        )
        assert (
            settings.get("threshold") == 10.0
        ), f"{layer.name}: threshold not copied from source layer"
    # The source layer's own settings remain unchanged.
    assert source_layer.metadata["settings"]["filter"]["size"] == 5


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


def test_docked_plotter_layout(make_napari_viewer, qtbot):
    """A plotter docked in the napari window: where its analysis docks go,
    the title bar strip it reserves, the height it claims, phasor centers
    raising the statistics dock, and hiding, floating and closing it."""
    from qtpy.QtCore import QSize, Qt

    viewer = make_napari_viewer()
    plotter = PlotterWidget(viewer)
    qtbot.addWidget(plotter)
    # Dock the plotter so the "split analysis below plotter" path executes.
    dock = viewer.window.add_dock_widget(
        plotter, name="Phasor Plot", area="right"
    )
    qt_window = viewer.window._qt_window

    # The reserved strip is the bar's real height less the height it claims.
    title_bar = dock.titleBarWidget()
    expected = max(0, title_bar.height() - title_bar.sizeHint().height())
    assert plotter._title_bar_overlap() == expected
    plotter._reserve_title_bar_overlap()
    assert plotter.layout().contentsMargins().top() == expected

    # A bar that reports nothing usable still gets the default strip, and no
    # title bar at all means no bar to clear.
    class _Unmeasurable:
        def sizeHint(self):
            return QSize(0, 0)

        def height(self):
            return 0

    with patch.object(dock, 'titleBarWidget', return_value=_Unmeasurable()):
        assert (
            plotter._title_bar_overlap()
            == PlotterWidget._TITLE_BAR_OVERLAP_FALLBACK
        )
    with patch.object(dock, 'titleBarWidget', return_value=None):
        assert plotter._title_bar_overlap() == 0

    # Fire the dock setup deterministically rather than via the init timer:
    # analysis docks in the right area (below the plotter), histogram and
    # statistics in the bottom area.
    plotter._analysis_dock_init_timer.stop()
    plotter._add_analysis_dock_widget()
    assert (
        qt_window.dockWidgetArea(plotter._analysis_dock)
        == Qt.RightDockWidgetArea
    )
    assert (
        qt_window.dockWidgetArea(plotter._histogram_dock)
        == Qt.BottomDockWidgetArea
    )
    assert (
        qt_window.dockWidgetArea(plotter._statistics_dock)
        == Qt.BottomDockWidgetArea
    )
    assert plotter._docks_initialized is True
    # The hosting dock is discoverable via the parent walk.
    assert plotter._find_plotter_dock() is not None
    # Resize path (right width + minimum bottom height) runs without error.
    plotter._resize_initial_docks()

    # Docked, the plotter claims only the height its aspect-locked plot
    # uses; the cap follows the panel width.
    assert plotter._spare_height_can_be_reused() is True
    plotter._resize_canvas_to_available_space()
    narrow_limit = plotter.maximumHeight()
    assert narrow_limit < PlotterWidget._NO_HEIGHT_LIMIT
    assert narrow_limit >= plotter.minimumHeight()
    # A wider panel fits a taller plot at the same aspect ratio...
    plotter._update_useful_height_limit(plotter.canvas_container.width() * 2)
    assert plotter.maximumHeight() > narrow_limit
    # ...and the widened limit schedules the deferred re-split, which hands
    # the plot the height it can use.
    assert plotter._claim_height_timer.isActive()
    plotter._claim_useful_height()
    # Closing the tabs leaves nobody to take the spare height.
    plotter._analysis_dock.hide()
    plotter._update_useful_height_limit(plotter.canvas_container.width())
    assert plotter.maximumHeight() == PlotterWidget._NO_HEIGHT_LIMIT
    plotter._analysis_dock.show()

    # The bar you drag the panel by must not cover the toolbar icons:
    # napari's QtCustomTitleBar reports a hard-coded 20 px size hint while
    # laying itself out taller, and QDockWidget puts the content at the
    # hinted height.
    qt_window.resize(1200, 900)
    # The overlap can only be measured once Qt has laid the dock out.
    qt_window.show()
    plotter._reserve_title_bar_overlap()
    qt_window.layout().activate()
    title_bar = dock.titleBarWidget()
    assert title_bar is not None

    def _bottom(widget):
        return widget.mapTo(qt_window, widget.rect().bottomLeft()).y() + 1

    def _top(widget):
        return widget.mapTo(qt_window, widget.rect().topLeft()).y()

    qtbot.waitUntil(
        lambda: _top(plotter.canvas_widget.toolbar) >= _bottom(title_bar)
    )

    # Laid out in a shown window, a plot entitled to more height than its
    # dock has takes it from the analysis tabs below. The re-split itself is
    # mocked: really resizing docks in a shown window crashed Windows CI.
    plotter_dock = plotter._find_plotter_dock()
    wanted_height = plotter_dock.height() + 100
    with (
        patch.object(plotter, 'maximumHeight', return_value=wanted_height),
        patch.object(qt_window, 'resizeDocks') as resize_docks,
    ):
        plotter._claim_useful_height()
    resize_docks.assert_called_once()
    docks, sizes, orientation = resize_docks.call_args[0]
    assert docks == [plotter_dock, plotter._analysis_dock]
    assert sizes[0] >= wanted_height
    assert orientation == Qt.Vertical
    # A plot already as tall as it can use leaves the split alone.
    with (
        patch.object(plotter, 'maximumHeight', return_value=0),
        patch.object(qt_window, 'resizeDocks') as resize_docks,
    ):
        plotter._claim_useful_height()
    resize_docks.assert_not_called()

    # Phasor centers: artist creation, statistics page switch and naming.
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)
    with patch.object(plotter._statistics_dock, 'raise_') as mock_raise:
        plotter.plotter_inputs_widget.phasor_center_checkbox.setChecked(True)
        assert len(plotter._phasor_center_artists) > 0
        # The Plot Settings tab shows the phasor center stats.
        plotter.tab_widget.setCurrentWidget(plotter.settings_tab)
        assert (
            plotter._statistics_stack.currentIndex()
            == plotter._phasor_center_stats_page_idx
        )
        # Toggling centers ON raises the statistics dock to visibility.
        mock_raise.assert_called()
    # A single layer's row is named after the layer, not 'Merged'.
    table = plotter._phasor_center_stats_widget._layer_table
    assert table.rowCount() == 1
    assert table.item(0, 0).text() == layer.name
    assert (
        layer.metadata.get('settings', {}).get('phasor_center_enabled') is True
    )
    plotter.plotter_inputs_widget.phasor_center_checkbox.setChecked(False)
    assert len(plotter._phasor_center_artists) == 0
    assert table.rowCount() == 0
    assert layer.metadata['settings']['phasor_center_enabled'] is False

    # napari's 'hide' button only hides the panel; nothing is closed. The
    # title bar's hide button is wired to QDockWidget.close(), which hides a
    # dock rather than destroying it.
    dock.title.hide_button.click()
    assert dock.isHidden()
    assert plotter._is_closing is False
    assert plotter._analysis_dock is not None
    assert plotter._histogram_dock is not None
    assert plotter._statistics_dock is not None
    assert "Phasor Plot" in viewer.window._wrapped_dock_widgets
    dock.show()

    # Undocking and re-docking the plotter must not close anything.
    dock.setFloating(True)
    dock.setFloating(False)
    assert plotter._is_closing is False
    assert plotter._analysis_dock is not None
    assert plotter._plotter_dock_ref is dock

    # Clicking close 'X' on the napari title bar closes associated docks.
    dock.title.close_button.click()
    assert plotter._is_closing is True
    assert plotter._analysis_dock is None
    assert plotter._histogram_dock is None
    assert plotter._statistics_dock is None


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


def test_phasor_centers_for_several_layers(make_viewer_model):
    """Merged mode gives a single 'Merged' center; individual mode one
    center and statistics row per selected layer."""
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)

    layer1 = create_image_layer_with_phasors()
    layer2 = create_image_layer_with_phasors()
    viewer.add_layer(layer1)
    viewer.add_layer(layer2)
    plotter.image_layers_checkable_combobox.setCheckedItems(
        [layer1.name, layer2.name]
    )
    plotter._process_layer_selection_change()
    layer_table = plotter._phasor_center_stats_widget._layer_table
    group_table = plotter._phasor_center_stats_widget._group_table

    plotter._phasor_center_display_mode = "Merged"
    plotter.plotter_inputs_widget.phasor_center_checkbox.setChecked(True)
    plotter._update_phasor_centers()
    assert len(plotter._phasor_center_artists) == 1
    assert layer_table.rowCount() == 1
    assert layer_table.item(0, 0).text() == "Merged"
    assert group_table.rowCount() == 0

    plotter._phasor_center_display_mode = "Individual layers"
    plotter._update_phasor_centers()
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


def test_phasor_center_grouped_median_uses_pooled_samples(make_viewer_model):
    """Grouped mode should use pooled samples with selected median method."""
    viewer = make_viewer_model()
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


def test_deferred_tab_updates(make_viewer_model):
    """The current analysis tab is restored at once; the others are marked
    to restore when shown, through whichever hook they provide."""
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)
    layer = create_image_layer_with_phasors()
    layer.name = "L1"
    viewer.add_layer(layer)

    # _run_deferred_tab_update restores a dirty current tab. Switch to it
    # FIRST (the tab-switch handler clears the flag), then mark it dirty.
    plotter.tab_widget.setCurrentWidget(plotter.components_tab)
    plotter.components_tab._needs_update = True
    with patch.object(
        plotter.components_tab, '_restore_on_layer_change'
    ) as mock_restore:
        plotter._run_deferred_tab_update(plotter.components_tab)
        mock_restore.assert_called_once()
    assert plotter.components_tab._needs_update is False

    # ...and is a no-op on a clean tab.
    with patch.object(
        plotter.components_tab, '_restore_on_layer_change'
    ) as mock_restore:
        plotter._run_deferred_tab_update(plotter.components_tab)
        mock_restore.assert_not_called()

    # A tab without _restore_on_layer_change falls back to
    # _on_image_layer_changed.
    class _Stub:
        _needs_update = True

        def __init__(self):
            self.called = False

        def _on_image_layer_changed(self):
            self.called = True

    stub = _Stub()
    original = plotter.components_tab
    plotter.components_tab = stub
    try:
        plotter._run_deferred_tab_update(stub)
        assert stub.called is True
        assert stub._needs_update is False
    finally:
        plotter.components_tab = original

    # _apply_layer_data restores whichever deferrable tab is current.
    for tab in (
        plotter.phasor_mapping_tab,
        plotter.components_tab,
        plotter.fret_tab,
    ):
        plotter.tab_widget.setCurrentWidget(tab)
        with patch.object(tab, '_restore_on_layer_change') as mock_restore:
            plotter._apply_layer_data(
                "L1", reset_zoom=False, sync_frequency=False
            )
            assert mock_restore.call_count >= 1

    # From a non-deferrable tab, all three deferrable tabs get marked.
    plotter.tab_widget.setCurrentWidget(plotter.filter_tab)
    plotter.phasor_mapping_tab._needs_update = False
    plotter.components_tab._needs_update = False
    plotter.fret_tab._needs_update = False
    plotter._apply_layer_data("L1", reset_zoom=False, sync_frequency=False)
    assert plotter.phasor_mapping_tab._needs_update is True
    assert plotter.components_tab._needs_update is True
    assert plotter.fret_tab._needs_update is True


def test_refresh_phasor_data_immediately_restores_current_deferrable_tab(
    make_viewer_model,
):
    """When a deferrable tab is current, refresh_phasor_data restores it now."""
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)

    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)

    # Make components_tab the current tab
    plotter.tab_widget.setCurrentWidget(plotter.components_tab)
    plotter.components_tab._needs_update = False

    with patch.object(
        plotter.components_tab, '_restore_on_layer_change'
    ) as mock_restore:
        plotter.refresh_phasor_data()
        mock_restore.assert_called_once()

    plotter.deleteLater()


def test_update_grid_view(make_viewer_model):
    """Grid mode and layer visibility follow the selection, without
    redundant visibility writes."""
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)

    layers = []
    for i in range(3):
        layer = create_image_layer_with_phasors()
        layer.name = f"layer_{i}"
        viewer.add_layer(layer)
        layers.append(layer)
    a, b, c = layers

    # With grid on and every layer visible, nothing is written.
    viewer.grid.enabled = True
    for layer in layers:
        layer.visible = True
    write_count = {'n': 0}
    for layer in layers:
        # Observe via the events.visible signal which fires only on change.
        layer.events.visible.connect(
            lambda e: write_count.__setitem__('n', write_count['n'] + 1)
        )
    plotter._update_grid_view(list(layers))
    assert (
        write_count['n'] == 0
    ), "Expected zero visibility writes when state already matches"

    # A single visible layer does not re-disable a disabled grid.
    viewer.grid.enabled = False
    plotter._update_grid_view([a])
    assert viewer.grid.enabled is False
    assert a.visible is True

    # The lone selected layer is made visible if it was hidden.
    a.visible = False
    plotter._update_grid_view([a])
    assert a.visible is True

    # Multi-layer selection enables grid mode and syncs analysis layers.
    comp_a = viewer.add_image(
        np.zeros((5, 5)), name=_lp_name("Component 1", "layer_0")
    )
    comp_c = viewer.add_image(
        np.zeros((5, 5)), name=_lp_name("Component 1", "layer_2")
    )
    comp_c.visible = True
    viewer.grid.enabled = False
    plotter._update_grid_view([a, b])
    assert viewer.grid.enabled is True
    assert a.visible is True
    assert b.visible is True
    assert comp_a.visible is True
    assert c.visible is False
    assert comp_c.visible is False


def test_update_layer_visibility_for_selection(make_viewer_model):
    """Selected intensity layers and their analysis layers are shown, the
    rest hidden; unrelated layers keep their visibility; an analysis layer
    belongs only to the intensity layer whose name it carries."""
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)

    a = create_image_layer_with_phasors()
    a.name = "imgA"
    viewer.add_layer(a)
    b = create_image_layer_with_phasors()
    b.name = "imgB"
    viewer.add_layer(b)

    # _is_phasor_intensity_layer detects intensity layers with phasor data.
    plain = viewer.add_image(np.zeros((5, 5)), name="plain")
    assert plotter._is_phasor_intensity_layer(a) is True
    assert plotter._is_phasor_intensity_layer(plain) is False

    # Analysis layers derived from each intensity layer.
    comp_a = viewer.add_image(
        np.zeros((5, 5)), name=_lp_name("Component 1", "imgA")
    )
    fret_b = viewer.add_image(np.zeros((5, 5)), name="imgB [FRET efficiency]")
    unrelated_visible = viewer.add_image(np.zeros((5, 5)), name="reference")
    unrelated_hidden = viewer.add_image(np.zeros((5, 5)), name="scratch")

    # Start from a mixed visibility state to exercise both directions.
    a.visible = False
    b.visible = True
    comp_a.visible = False
    fret_b.visible = True
    unrelated_visible.visible = True
    unrelated_hidden.visible = False

    plotter._update_layer_visibility_for_selection({"imgA"})
    assert a.visible is True
    assert comp_a.visible is True
    assert b.visible is False
    assert fret_b.visible is False
    # Layers not derived from any phasor layer keep their visibility.
    assert unrelated_visible.visible is True
    assert unrelated_hidden.visible is False

    # "other img" ends with "img", but its analysis layer is derived from
    # "other img" only.
    short = create_image_layer_with_phasors()
    short.name = "img"
    viewer.add_layer(short)
    long = create_image_layer_with_phasors()
    long.name = "other img"
    viewer.add_layer(long)
    analysis = viewer.add_image(
        np.zeros((5, 5)), name=_lp_name("Component 1", "other img")
    )
    analysis.visible = True
    plotter._update_layer_visibility_for_selection({"img"})
    assert short.visible is True
    assert long.visible is False
    assert analysis.visible is False
    plotter._update_layer_visibility_for_selection({"other img"})
    assert analysis.visible is True

    # Common harmonics skip a layer without harmonics, and a disjoint set
    # has none in common.
    no_harmonics = create_image_layer_with_phasors()
    no_harmonics.name = "no_harmonics"
    no_harmonics.metadata.pop("harmonics", None)
    viewer.add_layer(no_harmonics)
    result = plotter._get_common_harmonics([a, no_harmonics])
    assert result is not None
    expected = sorted(np.atleast_1d(a.metadata["harmonics"]).tolist())
    assert list(result) == expected
    layer1 = create_image_layer_with_phasors(harmonic=[1, 2])
    layer1.name = "L1"
    layer2 = create_image_layer_with_phasors(harmonic=[3, 4])
    layer2.name = "L2"
    viewer.add_layer(layer1)
    viewer.add_layer(layer2)
    assert plotter._get_common_harmonics([layer1, layer2]) is None


def test_features_cache_is_invalidated_by_every_mutation(make_viewer_model):
    """Masking, restoring and changing the harmonic invalidate the cached
    features; an unsupported mask layer changes nothing."""
    from napari.layers import Labels

    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)
    image_layer = create_image_layer_with_phasors()
    viewer.add_layer(image_layer)

    # An Image layer is neither Shapes nor (Labels with data.any()), so the
    # mask is not applied.
    other_image = create_image_layer_with_phasors()
    plotter._apply_mask_to_phasor_data(other_image, image_layer)
    assert "mask" not in image_layer.metadata

    # Invalidating clears both cache fields.
    plotter.get_merged_features()
    assert plotter._features_cache is not None
    assert plotter._features_cache_key is not None
    plotter._invalidate_features_cache()
    assert plotter._features_cache is None
    assert plotter._features_cache_key is None

    # _apply_mask_to_phasor_data invalidates the cache after mutating G/S.
    mask_layer = Labels(
        np.ones(image_layer.data.shape, dtype=np.uint8), name="mask"
    )
    viewer.add_layer(mask_layer)
    plotter.get_merged_features()
    sentinel = ('stale-mask-test',)
    plotter._features_cache = sentinel
    plotter._apply_mask_to_phasor_data(mask_layer, image_layer)
    assert plotter._features_cache is not sentinel

    # So does restoring the original phasor data...
    plotter.get_merged_features()
    sentinel = ('stale-restore-test',)
    plotter._features_cache = sentinel
    plotter._restore_original_phasor_data(image_layer)
    assert plotter._features_cache is not sentinel

    # ...and changing the harmonic.
    plotter.get_merged_features()
    sentinel = ('stale-harmonic-test',)
    plotter._features_cache = sentinel
    plotter._on_harmonic_changed(2)
    assert plotter._features_cache is not sentinel

    # Replacing a layer's G/S arrays (as a script would) is picked up
    # without invalidating anything; unchanged arrays keep the cache.
    cached = plotter.get_merged_features()
    assert plotter.get_merged_features() is cached
    image_layer.metadata['G'] = image_layer.metadata['G'] * 0.5
    replaced = plotter.get_merged_features()
    assert replaced is not cached
    np.testing.assert_allclose(np.sort(replaced[0]), np.sort(cached[0] * 0.5))

    # An array edited in place is the same object, so the cache needs
    # refresh_phasor_data() to notice it.
    image_layer.metadata['G'] *= 2
    assert plotter.get_merged_features() is replaced
    plotter.refresh_phasor_data()
    np.testing.assert_allclose(
        np.sort(plotter.get_merged_features()[0]), np.sort(cached[0])
    )


def test_on_mask_data_changed(make_viewer_model):
    """An edited mask re-applies only to the layers it is assigned to."""
    from napari.layers import Labels

    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)

    layer1 = create_image_layer_with_phasors()
    layer1.name = "img1"
    layer2 = create_image_layer_with_phasors()
    layer2.name = "img2"
    viewer.add_layer(layer1)
    viewer.add_layer(layer2)

    def event_from(mask):
        class _Event:
            source = mask

        return _Event()

    # Single-layer mode returns early if the mask is not the one selected.
    some_mask = Labels(
        np.ones(layer1.data.shape, dtype=np.uint8), name="some_mask"
    )
    viewer.add_layer(some_mask)
    plotter.image_layers_checkable_combobox.setCheckedItems(["img1"])
    plotter.mask_layer_combobox.setCurrentText("None")
    g_before = layer1.metadata['G'].copy()
    plotter._on_mask_data_changed(event_from(some_mask))
    np.testing.assert_array_equal(layer1.metadata['G'], g_before)

    # Multi-layer mode returns early if no layer is assigned to this mask.
    plotter.image_layers_checkable_combobox.setCheckedItems(["img1", "img2"])
    plotter._mask_assignments = {}
    plotter._on_mask_data_changed(event_from(some_mask))
    np.testing.assert_array_equal(layer1.metadata['G'], g_before)

    # Otherwise it filters by the per-layer assignment: a partial mask on
    # img1 only.
    mask_data = np.zeros(layer1.data.shape, dtype=np.uint8)
    mask_data[..., 0] = 1  # only first column is "inside" the mask
    mask = Labels(mask_data, name="mask_a")
    viewer.add_layer(mask)
    plotter._mask_assignments = {"img1": "mask_a"}
    g_before_layer2 = layer2.metadata['G'].copy()
    plotter._on_mask_data_changed(event_from(mask))
    np.testing.assert_array_equal(layer2.metadata['G'], g_before_layer2)
    assert np.isnan(layer1.metadata['G']).any()


def test_plot_settings_dialogs_and_view_helpers(make_viewer_model, qtbot):
    """The contour and phasor center settings dialogs, the shared group
    metadata they open with, 'Individual layers' rendering, scroll zoom,
    toolbar patches and the dock show helpers."""
    from qtpy.QtWidgets import QDialog

    import napari_phasors.plotter

    viewer = make_viewer_model()
    layer1 = create_image_layer_with_phasors()
    layer2 = create_image_layer_with_phasors()
    viewer.add_layer(layer1)
    viewer.add_layer(layer2)
    plotter = PlotterWidget(viewer)

    # Grouping stored on the layers wins over the plotter's stale copy:
    # groups are shared with the histogram tabs through each layer's
    # settings['group'] entry.
    for layer, group_name in ((layer1, "Ctrl"), (layer2, "Trt")):
        layer.metadata.setdefault('settings', {})['group'] = {
            'name': group_name,
            'color': [1.0, 0.0, 0.0],
        }
    plotter._contour_group_assignments = {layer1.name: 1, layer2.name: 1}
    plotter._contour_group_names = {1: "Everything"}
    captured = {}

    class CapturingDialog:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

        def exec(self):
            return QDialog.Rejected

    with (
        patch.object(
            plotter,
            "get_selected_layer_names",
            lambda: [layer1.name, layer2.name],
        ),
        patch.object(
            napari_phasors.plotter,
            "ContourLayerSettingsDialog",
            CapturingDialog,
        ),
    ):
        plotter._on_contour_layer_settings_clicked()
    assignments = captured["group_assignments"]
    names = captured["group_names"]
    assert names[assignments[layer1.name]] == "Ctrl"
    assert names[assignments[layer2.name]] == "Trt"

    # An accepted contour dialog applies its choices.
    class ContourDialog:
        def __init__(self, *args, **kwargs):
            pass

        def exec(self):
            return QDialog.Accepted

        def get_display_mode(self):
            return "individual"

        def get_merged_colormap(self):
            return "viridis"

        def get_merged_style(self):
            return "solid"

        def get_merged_color(self):
            return (1, 1, 1, 1)

        def get_show_legend(self):
            return True

        def get_layer_styles(self):
            return {}

        def get_group_styles(self):
            return {}

        def get_layer_colors(self):
            return {"Layer1": (1, 0, 0, 1)}

        def get_group_assignments(self):
            return {"Layer1": 1}

        def get_group_colors(self):
            return {1: (1, 0, 0, 1)}

        def get_group_names(self):
            return {1: "Group1"}

    # So does an accepted phasor center dialog.
    class CenterDialog:
        def __init__(self, *args, **kwargs):
            pass

        def exec(self):
            return QDialog.Accepted

        def get_display_mode(self):
            return "merged"

        def get_center_method(self):
            return "centroid"

        def get_marker_size(self):
            return 10

        def get_alpha(self):
            return 1.0

        def get_merged_color(self):
            return (0, 1, 0, 1)

        def get_merged_marker(self):
            return "x"

        def get_show_legend(self):
            return False

        def get_layer_colors(self):
            return {}

        def get_layer_markers(self):
            return {}

        def get_group_assignments(self):
            return {}

        def get_group_colors(self):
            return {}

        def get_group_names(self):
            return {}

        def get_group_markers(self):
            return {}

    with (
        patch.object(
            plotter, "get_selected_layer_names", lambda: ["Layer1", "Layer2"]
        ),
        patch.object(
            napari_phasors.plotter, "ContourLayerSettingsDialog", ContourDialog
        ),
        patch.object(
            napari_phasors.plotter,
            "PhasorCenterLayerSettingsDialog",
            CenterDialog,
        ),
    ):
        plotter._on_contour_layer_settings_clicked()
        assert plotter._contour_display_mode == "individual"
        assert plotter._contour_layer_colors == {"Layer1": (1, 0, 0, 1)}
        plotter._on_phasor_center_configure_clicked()
        assert plotter._phasor_center_display_mode == "merged"
        assert plotter._phasor_center_color == (0, 1, 0, 1)

    # 'Individual layers' contour and histogram rendering.
    x_data = np.array([1.0, 2.0, 5.0, 6.0])
    y_data = np.array([3.0, 4.0, 7.0, 8.0])
    layer_feature_map = {
        "Layer1": (np.array([1.0, 2.0]), np.array([3.0, 4.0])),
        "Layer2": (np.array([5.0, 6.0]), np.array([7.0, 8.0])),
    }
    with patch.object(
        plotter, "_get_selected_layer_feature_map", lambda: layer_feature_map
    ):
        plotter._contour_display_mode = "Individual layers"
        plotter._update_contour_plot(x_data, y_data)
        assert len(plotter._contour_collections) > 0
        axes = plotter.canvas_widget.axes
        n_artists_before = len(axes.collections) + len(axes.images)
        plotter._histogram_display_mode = "Individual layers"
        plotter._update_histogram_plot(x_data, y_data)
        n_artists_after = len(axes.collections) + len(axes.images)
        assert n_artists_after >= n_artists_before

    # Scroll-wheel zoom shrinks the axis limits around the cursor.
    ax = plotter.canvas_widget.axes
    event = MagicMock()
    event.inaxes = ax
    event.step = 1
    event.xdata, event.ydata = 0.5, 0.5
    orig_xlim = ax.get_xlim()
    plotter._on_scroll_zoom(event)
    assert ax.get_xlim() != orig_xlim

    # Toolbar release patches and home are callable without error.
    toolbar = getattr(plotter.canvas_widget, "toolbar", None)
    if toolbar:
        ev = MagicMock()
        if hasattr(toolbar, "release_zoom"):
            toolbar.release_zoom(ev)
        if hasattr(toolbar, "release_pan"):
            toolbar.release_pan(ev)
        if hasattr(toolbar, "home"):
            toolbar.home()

    # Dock helpers work both without docks and with dock objects present.
    plotter._show_statistics_dock()
    plotter._show_analysis_dock()
    plotter._show_histogram_dock()
    plotter._statistics_dock = MagicMock()
    plotter._analysis_dock = MagicMock()
    plotter._histogram_dock = MagicMock()
    plotter._show_statistics_dock()
    plotter._show_analysis_dock()
    plotter._show_histogram_dock()
    assert plotter._statistics_dock.setVisible.called


def _plotter_with_grouped_centers(make_viewer_model, assignments):
    """Return a plotter showing grouped phasor centers for three layers."""
    from napari_phasors._tests.test_plotter import (
        create_image_layer_with_phasors,
    )
    from napari_phasors.plotter import PlotterWidget

    viewer = make_viewer_model()
    for index in range(3):
        layer = create_image_layer_with_phasors()
        layer.name = f"img{index}"
        viewer.add_layer(layer)

    plotter = PlotterWidget(viewer)
    plotter.image_layers_checkable_combobox.setCheckedItems(
        ["img0", "img1", "img2"]
    )
    plotter.image_layers_checkable_combobox.setPrimaryLayer("img0")
    plotter._layer_selection_timer.stop()
    plotter._process_layer_selection_change()

    plotter._phasor_center_display_mode = "Grouped"
    plotter._phasor_center_group_assignments = dict(assignments)
    plotter._phasor_center_group_names = {1: "A", 2: "B"}
    for key in (
        "phasor_center_display_mode",
        "phasor_center_group_assignments",
        "phasor_center_group_names",
    ):
        plotter._update_setting_in_metadata(key, getattr(plotter, f"_{key}"))
    plotter._on_phasor_center_toggled(True)
    return plotter


def _reselect(plotter, names):
    """Check *names* and run the debounced selection handler immediately."""
    plotter.image_layers_checkable_combobox.setCheckedItems(names)
    plotter._layer_selection_timer.stop()
    plotter._process_layer_selection_change()


def test_phasor_center_group_dot_drops_when_its_layers_are_unchecked(
    make_viewer_model,
):
    """A group with no selected layer left loses its center dot."""
    plotter = _plotter_with_grouped_centers(
        make_viewer_model, {"img0": 2, "img1": 1, "img2": 1}
    )
    assert len(plotter._phasor_center_artists) == 2

    # Uncheck both layers of group A; the primary (group B) stays selected.
    _reselect(plotter, ["img0"])

    assert plotter._phasor_center_enabled is True
    assert len(plotter._phasor_center_artists) == 1

    _reselect(plotter, ["img0", "img1", "img2"])
    assert len(plotter._phasor_center_artists) == 2

    plotter.deleteLater()


def test_phasor_center_dots_cleared_when_selection_change_disables_them(
    make_viewer_model,
):
    """Dots never outlive the selection that produced them.

    Unchecking the primary layer restores the settings of the new primary,
    which may have phasor centers switched off. The dots drawn for the old
    selection have to go with them.
    """
    plotter = _plotter_with_grouped_centers(
        make_viewer_model, {"img0": 1, "img1": 1, "img2": 2}
    )
    assert len(plotter._phasor_center_artists) == 2
    # Switching the centers on stored them in every selected layer; img2 is
    # given its own settings with them off.
    plotter.viewer.layers["img2"].metadata["settings"][
        "phasor_center_enabled"
    ] = False

    # Group A holds the primary layer, so unchecking it swaps the primary.
    _reselect(plotter, ["img2"])
    assert plotter._phasor_center_artists == []

    # And unchecking everything leaves nothing behind either.
    _reselect(plotter, [])
    assert plotter._phasor_center_artists == []

    plotter.deleteLater()


def test_plot_style_callbacks(make_viewer_model, qtbot, monkeypatch):
    """Plot colors, marker and contour colors, colormap changes, lifetime
    ticks and artist-signal teardown."""
    import matplotlib.pyplot as plt
    import qtpy.QtWidgets
    from matplotlib.text import Text
    from qtpy.QtGui import QColor

    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)

    # Saving, overriding and restoring plot colors round-trips.
    saved_colors = plotter._capture_plot_colors()
    assert isinstance(saved_colors, dict)
    assert "axes" in saved_colors
    plotter._apply_plot_colors("red")
    plotter._apply_plot_colors_from_saved(saved_colors)

    # The marker colour picked from the dialog is stored.
    mock_color = MagicMock()
    mock_color.isValid.return_value = True
    mock_color.name.return_value = "#ff0000"
    with patch.object(
        qtpy.QtWidgets.QColorDialog,
        "getColor",
        lambda *args, **kwargs: mock_color,
    ):
        scatter = plotter.canvas_widget.artists['SCATTER']
        scatter.data = np.array([[0.5, 0.5], [0.6, 0.6]])
        plotter._on_marker_color_clicked()
    assert plotter._marker_color == "#ff0000"
    assert scatter.color == "#ff0000"
    sc = scatter._mpl_artists.get("scatter")
    assert sc is not None
    np.testing.assert_allclose(
        sc.get_facecolors()[0][:3], [1.0, 0.0, 0.0], atol=1e-3
    )
    assert len(sc.get_edgecolors()) == 0
    assert sc.get_linewidths()[0] == 0

    # The 'Select color...' sentinel switches the histogram style to solid.
    plotter.plotter_inputs_widget.colormap_combobox.setCurrentText(
        "Select color..."
    )
    plotter._on_colormap_changed()
    assert plotter._histogram_style == "solid"

    # White-background toggle restyles without error.
    plotter.on_white_background_changed()

    # Colormap changes for a single-layer contour and the histogram.
    with (
        patch.object(plotter, "get_selected_layer_names", lambda: ["Layer1"]),
        patch.object(plotter, 'plot', lambda: None),
        patch.object(plotter, 'refresh_current_plot', lambda: None),
    ):
        plotter.plot_type = 'CONTOUR'
        plotter.plotter_inputs_widget.colormap_combobox.setCurrentText(
            "Select color..."
        )
        plotter._on_colormap_changed()
        assert plotter._single_contour_style == 'solid'
        plotter.plotter_inputs_widget.colormap_combobox.setCurrentText(
            "viridis"
        )
        plotter._on_colormap_changed()
        assert plotter._single_contour_style == 'colormap'
        assert plotter._single_contour_colormap == "viridis"
        plotter.plot_type = 'HISTOGRAM2D'
        plotter.plotter_inputs_widget.colormap_combobox.setCurrentText(
            "plasma"
        )
        plotter._on_colormap_changed()
        assert plotter.histogram_colormap == "plasma"

    # The single contour colour, in HISTOGRAM2D and CONTOUR modes.
    class MockColorDialog:
        @staticmethod
        def getColor(parent=None):
            return QColor(0, 255, 0)

    monkeypatch.setattr(qtpy.QtWidgets, "QColorDialog", MockColorDialog)
    with (
        patch.object(plotter, 'plot', lambda: None),
        patch.object(plotter, 'refresh_current_plot', lambda: None),
    ):
        plotter.plot_type = 'HISTOGRAM2D'
        plotter._on_single_contour_color_clicked()
        assert plotter._histogram_style == 'solid'
        assert plotter._histogram_color == (0.0, 1.0, 0.0)
        plotter.plot_type = 'CONTOUR'
        plotter._on_single_contour_color_clicked()
        assert plotter._single_contour_style == 'solid'
        assert plotter._single_contour_color == (0.0, 1.0, 0.0)
    monkeypatch.undo()

    # The tick size control shows in semicircle mode when the layer has a
    # frequency.
    piw = plotter.plotter_inputs_widget
    frequency = {'value': None}
    with patch.object(
        plotter, "_get_frequency_from_layer", lambda: frequency['value']
    ):
        plotter._update_semi_circle_plot(plotter.canvas_widget.axes)
        assert piw.lifetime_tick_size_spinbox.isHidden()
        assert piw.label_lifetime_tick_size.isHidden()
        frequency['value'] = 80.0
        plotter._update_semi_circle_plot(plotter.canvas_widget.axes)
        assert not piw.lifetime_tick_size_spinbox.isHidden()
        assert not piw.label_lifetime_tick_size.isHidden()
        plotter.toggle_semi_circle = False
        assert piw.lifetime_tick_size_spinbox.isHidden()
        plotter.toggle_semi_circle = True
        assert not piw.lifetime_tick_size_spinbox.isHidden()

    # The tick size scales the labels, never the tick marks.
    with patch.object(plotter, "_get_frequency_from_layer", lambda: 80.0):
        fig, ax = plt.subplots()

        def tick_artists():
            plotter.semi_circle_plot_artist_list.clear()
            plotter._add_lifetime_ticks_to_semicircle(ax)
            artists = plotter.semi_circle_plot_artist_list
            lines = [a for a in artists if not isinstance(a, Text)]
            labels = [a for a in artists if isinstance(a, Text)]
            return lines[1], labels[1]

        line, label = tick_artists()
        width, fontsize = line.get_linewidth(), label.get_fontsize()
        tick_data = line.get_xydata().copy()
        label_position = label.get_position()
        plotter._updating_settings = True
        piw.lifetime_tick_size_spinbox.setValue(2.0)
        plotter._updating_settings = False
        line, label = tick_artists()
        assert line.get_linewidth() == pytest.approx(width)
        np.testing.assert_allclose(line.get_xydata(), tick_data)
        assert label.get_position() == pytest.approx(label_position)
        assert label.get_fontsize() == pytest.approx(2 * fontsize)
        plt.close(fig)

        # The semicircle ticks draw against a narrow canvas too.
        fig, ax = plt.subplots()
        with patch.object(
            plotter,
            "canvas_widget",
            type('MockCanvas', (), {'width': lambda *args, **kwargs: 300})(),
        ):
            plotter._add_lifetime_ticks_to_semicircle(
                ax, visible=True, alpha=1.0, zorder=10
            )
        assert len(plotter.semi_circle_plot_artist_list) > 0
        plt.close(fig)

    # Signal teardown handles non-image layers (shapes/labels) gracefully.
    viewer.add_shapes(np.random.random((2, 2)))
    viewer.add_labels(np.random.randint(0, 2, (10, 10)))
    plotter._disconnect_all_artist_signals()


def test_get_masked_gs(make_viewer_model):
    from napari_phasors.plotter import PlotterWidget

    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)

    # Not having phasor data
    res = plotter.get_masked_gs()
    assert res == (None, None)
    res = plotter.get_masked_gs(return_valid_mask=True)
    assert res == (None, None, None)

    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)
    plotter.image_layers_checkable_combobox.setCheckedItems([layer.name])
    plotter._process_layer_selection_change()

    # Test valid
    g, s = plotter.get_masked_gs()
    assert g is not None
    assert s is not None

    g, s, valid = plotter.get_masked_gs(return_valid_mask=True)
    assert valid is not None

    g, s = plotter.get_masked_gs(flat=True)
    assert g.ndim == 1

    g, s, valid = plotter.get_masked_gs(flat=True, return_valid_mask=True)
    assert valid.ndim == 1

    # Invalid harmonic
    res = plotter.get_masked_gs(harmonic=999)
    assert res == (None, None)


def test_get_masked_gs_single_harmonic_layer(make_viewer_model):
    """A layer with a single harmonic (no leading harmonic axis in G/S)
    should return G/S as-is, and its spatial shape should match the image,
    not be trimmed as if a harmonic axis existed."""
    from napari_phasors.plotter import PlotterWidget

    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)

    layer = create_image_layer_with_phasors(harmonic=1)
    assert layer.metadata["G"].ndim == layer.data.ndim
    viewer.add_layer(layer)
    plotter.image_layers_checkable_combobox.setCheckedItems([layer.name])
    plotter._process_layer_selection_change()

    assert plotter.get_phasor_spatial_shape() == layer.data.shape

    g, s = plotter.get_masked_gs()
    assert g.shape == layer.data.shape
    assert s.shape == layer.data.shape
    np.testing.assert_array_equal(g, layer.metadata["G"])
    np.testing.assert_array_equal(s, layer.metadata["S"])


def test_closing_a_plotter_closes_its_docks(make_napari_viewer, qtbot):
    """Every way of closing a plotter takes its associated docks (and the
    viewer's bottom corners) with it; a destroyed analysis dock is re-added."""
    from qtpy.QtCore import Qt

    viewer = make_napari_viewer()
    qt_window = viewer.window._qt_window

    def new_plotter(docked=True, analysis=True):
        plotter = PlotterWidget(viewer)
        qtbot.addWidget(plotter)
        plotter._analysis_dock_init_timer.stop()
        dock = None
        if docked:
            dock = viewer.window.add_dock_widget(
                plotter, name="Phasor Plot", area="right"
            )
        if analysis:
            plotter._add_analysis_dock_widget()
        return plotter, dock

    def assert_closed(plotter):
        assert plotter._is_closing is True
        assert plotter._analysis_dock is None
        assert plotter._histogram_dock is None
        assert plotter._statistics_dock is None

    # Docking hands the bottom corners to the side areas; closing restores
    # the viewer's original corner ownership.
    original_left = qt_window.corner(Qt.BottomLeftCorner)
    original_right = qt_window.corner(Qt.BottomRightCorner)
    plotter, _ = new_plotter(docked=False)
    assert qt_window.corner(Qt.BottomLeftCorner) == Qt.LeftDockWidgetArea
    assert qt_window.corner(Qt.BottomRightCorner) == Qt.RightDockWidgetArea
    assert (
        plotter._original_bottom_corners[Qt.BottomLeftCorner] == original_left
    )
    assert (
        plotter._original_bottom_corners[Qt.BottomRightCorner]
        == original_right
    )
    # Capture is idempotent: reassigning again must not overwrite the stored
    # originals with the already-reassigned values.
    plotter._assign_bottom_corners_to_side_docks()
    assert (
        plotter._original_bottom_corners[Qt.BottomLeftCorner] == original_left
    )
    plotter.close()
    assert qt_window.corner(Qt.BottomLeftCorner) == original_left
    assert qt_window.corner(Qt.BottomRightCorner) == original_right

    # Removing the plotter's dock closes all associated window widgets.
    plotter, _ = new_plotter()
    assert plotter._analysis_dock is not None
    assert plotter._histogram_dock is not None
    assert plotter._statistics_dock is not None
    viewer.window.remove_dock_widget(plotter)
    assert_closed(plotter)
    assert viewer.window._wrapped_dock_widgets == {}

    # Calling plotter.close() closes the associated docks and, so no empty
    # panel is left behind, the plotter's own dock too.
    plotter, _ = new_plotter()
    plotter.close()
    assert_closed(plotter)
    assert viewer.window._wrapped_dock_widgets == {}

    # Closing an associated dock before the plotter preserves the re-open
    # button, and closing the plotter then closes the remaining docks.
    plotter, dock = new_plotter()
    plotter._histogram_dock.close()
    plotter._check_dock_visibility()
    assert not plotter.show_histogram_button.isHidden()
    assert not plotter._dock_buttons_widget.isHidden()
    assert plotter._is_closing is False
    dock.title.close_button.click()
    assert_closed(plotter)

    # Re-opening a destroyed analysis dock re-adds it to the right area.
    plotter, _ = new_plotter(analysis=False)
    destroyed = MagicMock()
    destroyed.setVisible.side_effect = RuntimeError("wrapped object deleted")
    plotter._analysis_dock = destroyed
    plotter._show_analysis_dock()
    assert (
        qt_window.dockWidgetArea(plotter._analysis_dock)
        == Qt.RightDockWidgetArea
    )
    plotter.close()


def test_phasor_center_grouped_skips_unassigned_layer(
    make_viewer_model, monkeypatch
):
    """An unassigned layer must not be pooled into the first group's center."""
    from napari.utils import notifications

    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)

    layer1 = create_image_layer_with_phasors(harmonic=[1])
    layer2 = create_image_layer_with_phasors(harmonic=[1])
    viewer.add_layer(layer1)
    viewer.add_layer(layer2)

    shape = layer1.data.shape
    _set_layer_harmonic0_samples(
        layer1,
        np.zeros(shape, dtype=float),
        np.zeros(shape, dtype=float),
        np.ones(shape, dtype=float),
    )
    _set_layer_harmonic0_samples(
        layer2,
        np.full(shape, 0.9, dtype=float),
        np.full(shape, 0.3, dtype=float),
        np.ones(shape, dtype=float),
    )

    plotter.image_layers_checkable_combobox.setCheckedItems(
        [layer1.name, layer2.name]
    )
    plotter._process_layer_selection_change()

    warnings = []
    monkeypatch.setattr(
        notifications, "show_warning", lambda msg: warnings.append(msg)
    )

    plotter._phasor_center_display_mode = "Grouped"
    # layer2 belongs to no group.
    plotter._phasor_center_group_assignments = {layer1.name: 1}
    plotter._phasor_center_group_names = {1: "Only group"}

    plotter.plotter_inputs_widget.phasor_center_checkbox.setChecked(True)
    plotter._update_phasor_centers()

    group_rows = _table_rows_by_name(
        plotter._phasor_center_stats_widget._group_table
    )
    assert set(group_rows) == {"Only group"}
    assert len(plotter._phasor_center_artists) == 1

    # The group centre is layer1's own centre, untouched by layer2.
    c1 = plotter._compute_single_center(layer1)
    assert c1 is not None
    np.testing.assert_allclose(group_rows["Only group"][:2], c1, atol=1e-6)

    assert len(warnings) == 1
    assert layer2.name in warnings[0]

    plotter.deleteLater()


def _grouped_dialog_rows(dialog):
    """Return ``{group_name: [checked layer, ...]}`` from a settings dialog."""
    return {
        row["name_edit"].text(): list(row["layer_combo"].checkedItems())
        for row in dialog._group_row_data
    }


def test_phasor_center_dialog_does_not_autocheck_unassigned(qtbot):
    """Opening the dialog must not tick an unassigned layer into group 1."""
    dialog = PhasorCenterLayerSettingsDialog(
        display_mode="Grouped",
        layer_labels=["A", "B"],
        group_assignments={"A": 1},
        group_names={1: "G1"},
    )
    qtbot.addWidget(dialog)

    assert _grouped_dialog_rows(dialog) == {"G1": ["A"]}
    assert dialog.get_group_assignments() == {"A": 1}
    assert dialog.get_unassigned_layers() == ["B"]


def test_contour_dialog_does_not_autocheck_unassigned(qtbot):
    """The contour dialog leaves unassigned layers unchecked as well."""
    from napari_phasors.plotter import ContourLayerSettingsDialog

    dialog = ContourLayerSettingsDialog(
        display_mode="Grouped",
        layer_labels=["A", "B", "C"],
        group_assignments={"A": 1, "C": 2},
        group_names={1: "G1", 2: "G2"},
    )
    qtbot.addWidget(dialog)

    assert _grouped_dialog_rows(dialog) == {"G1": ["A"], "G2": ["C"]}
    assert dialog.get_unassigned_layers() == ["B"]


def test_grouping_dialogs_warn_about_unassigned_layers(qtbot, monkeypatch):
    """Both grouping dialogs prompt before excluding unticked layers."""
    from qtpy.QtWidgets import QDialog, QMessageBox

    from napari_phasors.plotter import ContourLayerSettingsDialog

    def make_exec(role):
        def fake_exec(self):
            self._clicked = next(
                btn for btn in self.buttons() if self.buttonRole(btn) == role
            )

        return fake_exec

    monkeypatch.setattr(
        QMessageBox, "clickedButton", lambda self: self._clicked
    )

    for factory in (
        PhasorCenterLayerSettingsDialog,
        ContourLayerSettingsDialog,
    ):
        dialog = factory(
            display_mode="Grouped",
            layer_labels=["A", "B"],
            group_assignments={"A": 1},
            group_names={1: "G1"},
        )
        qtbot.addWidget(dialog)

        # "Go back" keeps the dialog open.
        monkeypatch.setattr(
            QMessageBox, "exec", make_exec(QMessageBox.RejectRole)
        )
        dialog.accept()
        assert dialog.result() != QDialog.Accepted

        # "Exclude them" confirms.
        monkeypatch.setattr(
            QMessageBox, "exec", make_exec(QMessageBox.AcceptRole)
        )
        dialog.accept()
        assert dialog.result() == QDialog.Accepted


def test_grouping_dialogs_no_warning_when_all_assigned(qtbot, monkeypatch):
    """No prompt when every layer belongs to a group."""
    from qtpy.QtWidgets import QDialog, QMessageBox

    from napari_phasors.plotter import ContourLayerSettingsDialog

    def fail_exec(self):
        raise AssertionError("no warning expected")

    monkeypatch.setattr(QMessageBox, "exec", fail_exec)

    for factory in (
        PhasorCenterLayerSettingsDialog,
        ContourLayerSettingsDialog,
    ):
        dialog = factory(
            display_mode="Grouped",
            layer_labels=["A", "B"],
            group_assignments={"A": 1, "B": 2},
            group_names={1: "G1", 2: "G2"},
        )
        qtbot.addWidget(dialog)
        dialog.accept()
        assert dialog.result() == QDialog.Accepted


def test_grouping_dialogs_enforce_one_group_per_layer(qtbot):
    """Contour and phasor-center rows offer each layer to one group only."""
    from napari_phasors.plotter import ContourLayerSettingsDialog

    for factory in (
        PhasorCenterLayerSettingsDialog,
        ContourLayerSettingsDialog,
    ):
        dialog = factory(
            display_mode="Grouped",
            layer_labels=["A", "B", "C"],
            group_assignments={"A": 1, "B": 2},
            group_names={1: "G1", 2: "G2"},
        )
        qtbot.addWidget(dialog)

        row1, row2 = dialog._group_row_data
        assert row1["layer_combo"].visibleItems() == ["A", "C"]
        assert row2["layer_combo"].visibleItems() == ["B", "C"]

        # Claiming the free layer removes it from the other row.
        row1["layer_combo"].setCheckedItems(["A", "C"])
        assert row2["layer_combo"].visibleItems() == ["B"]
        assert dialog.get_group_assignments() == {"A": 1, "C": 1, "B": 2}
        assert dialog.get_unassigned_layers() == []

        # Removing a group frees its layers for the remaining one.
        dialog._on_remove_group(row2["container"])
        assert row1["layer_combo"].visibleItems() == ["A", "B", "C"]
        assert dialog.get_group_assignments() == {"A": 1, "C": 1}
