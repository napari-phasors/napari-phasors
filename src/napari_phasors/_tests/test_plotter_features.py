from unittest.mock import MagicMock, patch

import numpy as np

from napari_phasors._tests.test_plotter import (  # noqa: E501
    create_image_layer_with_phasors,
)
from napari_phasors.plotter import (
    PhasorCenterLayerSettingsDialog,
    PlotterWidget,
)


def test_canvas_cleared_when_no_layer_selected(make_viewer_model):
    """Test that canvas phasor data is cleared but semicircle/circle remains when no layer is selected."""
    viewer = make_viewer_model()
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


def test_canvas_cleared_then_restored_with_new_layer(make_viewer_model):
    """Test that canvas can be restored after being cleared."""
    viewer = make_viewer_model()
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


def test_artist_data_cleared_when_no_layer_selected(make_viewer_model):
    """Test that biaplotter artists' internal data is properly cleared."""
    viewer = make_viewer_model()
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
    make_viewer_model,
):
    """Test that the import settings filter button applies to all selected layers."""
    viewer = make_viewer_model()
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
    make_viewer_model,
):
    """Test that the apply calibration if needed button applies to all selected layers."""
    viewer = make_viewer_model()
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
    make_viewer_model,
):
    """Test that the copy metadata from layer button applies to all selected layers."""
    viewer = make_viewer_model()
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


def test_phasor_center_multi_layer_merged_stats_name(make_viewer_model):
    """Multi-layer merged mode should produce a single 'Merged' row."""
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


def test_phasor_center_multi_layer_individual_stats(make_viewer_model):
    """Individual mode should create one center/stat row per selected layer."""
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


def test_run_deferred_tab_update_restore_branch(make_viewer_model):
    """_run_deferred_tab_update calls _restore_on_layer_change on dirty current tab."""
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)

    # Switch to components_tab FIRST (this triggers tab change handlers),
    # then set the dirty flag. Otherwise the tab-switch handler clears it
    # before our patch is in place.
    plotter.tab_widget.setCurrentWidget(plotter.components_tab)
    plotter.components_tab._needs_update = True

    with patch.object(
        plotter.components_tab, '_restore_on_layer_change'
    ) as mock_restore:
        plotter._run_deferred_tab_update(plotter.components_tab)
        mock_restore.assert_called_once()

    # After the run, the flag should be cleared.
    assert plotter.components_tab._needs_update is False

    plotter.deleteLater()


def test_run_deferred_tab_update_fallback_on_image_layer_changed(
    make_viewer_model,
):
    """_run_deferred_tab_update falls back to _on_image_layer_changed if no _restore."""
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)

    # Replace components_tab with a stub that has no _restore_on_layer_change.
    class _Stub:
        _needs_update = True

        def __init__(self):
            self.called = False

        def _on_image_layer_changed(self):
            self.called = True

    stub = _Stub()
    # The real components_tab attribute must NOT have _restore — temporarily
    # delete attribute so hasattr returns False.
    original = plotter.components_tab
    plotter.components_tab = stub
    try:
        plotter._run_deferred_tab_update(stub)
        assert stub.called is True
        assert stub._needs_update is False
    finally:
        plotter.components_tab = original

    plotter.deleteLater()


def test_run_deferred_tab_update_skips_clean_tab(make_viewer_model):
    """_run_deferred_tab_update is a no-op when _needs_update is False."""
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)

    plotter.components_tab._needs_update = False

    with patch.object(
        plotter.components_tab, '_restore_on_layer_change'
    ) as mock_restore:
        plotter._run_deferred_tab_update(plotter.components_tab)
        mock_restore.assert_not_called()

    plotter.deleteLater()


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


def test_apply_layer_data_with_no_layer_uses_polar_when_not_semicircle(
    make_viewer_model,
):
    """_apply_layer_data with no layer + toggle_semi_circle=False uses polar plot."""
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)

    plotter.toggle_semi_circle = False
    plotter._apply_layer_data("", reset_zoom=True, sync_frequency=False)

    assert plotter._g_array is None
    plotter.deleteLater()


def test_update_grid_view_skips_redundant_visibility_writes(
    make_viewer_model,
):
    """_update_grid_view does not write layer.visible when already correct."""
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)

    layers = []
    for i in range(3):
        layer = create_image_layer_with_phasors()
        layer.name = f"layer_{i}"
        viewer.add_layer(layer)
        layers.append(layer)

    selected_layers = list(layers)

    # Prime: enable grid and set all visible
    viewer.grid.enabled = True
    for layer in layers:
        layer.visible = True

    # Counter for visibility setter invocations
    write_count = {'n': 0}

    def make_setter(layer):
        # Observe via the events.visible signal which fires only on change.
        layer.events.visible.connect(
            lambda e: write_count.__setitem__('n', write_count['n'] + 1)
        )

    for layer in layers:
        make_setter(layer)

    # Call _update_grid_view — since everything matches, no events should fire
    plotter._update_grid_view(selected_layers)

    assert (
        write_count['n'] == 0
    ), "Expected zero visibility writes when state already matches"

    plotter.deleteLater()


def test_update_grid_view_single_layer_does_not_redisable_grid(
    make_viewer_model,
):
    """_update_grid_view skips disabling grid if already disabled."""
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)

    layer = create_image_layer_with_phasors()
    layer.name = "L1"
    viewer.add_layer(layer)

    viewer.grid.enabled = False
    layer.visible = True

    # Should not toggle grid or visibility because state already matches
    plotter._update_grid_view([layer])
    assert viewer.grid.enabled is False
    assert layer.visible is True

    plotter.deleteLater()


def test_update_grid_view_single_layer_makes_invisible_visible(
    make_viewer_model,
):
    """_update_grid_view makes the lone selected layer visible if it was hidden."""
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)

    layer = create_image_layer_with_phasors()
    layer.name = "L1"
    viewer.add_layer(layer)

    viewer.grid.enabled = False
    layer.visible = False  # simulate hidden state

    plotter._update_grid_view([layer])
    assert layer.visible is True

    plotter.deleteLater()


def test_is_phasor_intensity_layer(make_viewer_model):
    """_is_phasor_intensity_layer detects intensity layers with phasor data."""
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)

    intensity = create_image_layer_with_phasors()
    intensity.name = "imgA"
    viewer.add_layer(intensity)

    # Plain image layer without phasor metadata.
    plain = viewer.add_image(np.zeros((5, 5)), name="plain")

    assert plotter._is_phasor_intensity_layer(intensity) is True
    assert plotter._is_phasor_intensity_layer(plain) is False

    plotter.deleteLater()


def test_update_layer_visibility_shows_selected_and_associated(
    make_viewer_model,
):
    """Selected intensity layer and its analysis layers are made visible;
    non-selected layers and their analysis layers are hidden."""
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)

    a = create_image_layer_with_phasors()
    a.name = "imgA"
    viewer.add_layer(a)
    b = create_image_layer_with_phasors()
    b.name = "imgB"
    viewer.add_layer(b)

    # Analysis layers derived from each intensity layer.
    comp_a = viewer.add_image(
        np.zeros((5, 5)), name="Component 1 fractions: imgA"
    )
    fret_b = viewer.add_image(np.zeros((5, 5)), name="FRET efficiency: imgB")

    # Start from a mixed visibility state to exercise both directions.
    a.visible = False
    b.visible = True
    comp_a.visible = False
    fret_b.visible = True

    plotter._update_layer_visibility_for_selection({"imgA"})

    assert a.visible is True
    assert comp_a.visible is True
    assert b.visible is False
    assert fret_b.visible is False

    plotter.deleteLater()


def test_update_layer_visibility_leaves_unrelated_layers_untouched(
    make_viewer_model,
):
    """Layers not derived from any phasor layer keep their visibility."""
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)

    a = create_image_layer_with_phasors()
    a.name = "imgA"
    viewer.add_layer(a)

    unrelated_visible = viewer.add_image(np.zeros((5, 5)), name="reference")
    unrelated_visible.visible = True
    unrelated_hidden = viewer.add_image(np.zeros((5, 5)), name="scratch")
    unrelated_hidden.visible = False

    plotter._update_layer_visibility_for_selection({"imgA"})

    assert unrelated_visible.visible is True
    assert unrelated_hidden.visible is False

    plotter.deleteLater()


def test_update_layer_visibility_longest_suffix_wins(make_viewer_model):
    """When one intensity name is a suffix of another, the most specific
    (longest) intensity name claims the analysis layer."""
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)

    short = create_image_layer_with_phasors()
    short.name = "img"
    viewer.add_layer(short)
    long = create_image_layer_with_phasors()
    long.name = "other img"
    viewer.add_layer(long)

    # Name ends with ": other img" which contains ": img" as well; the
    # longer intensity name must win so this is associated with "other img".
    analysis = viewer.add_image(
        np.zeros((5, 5)), name="Component 1 fractions: other img"
    )
    analysis.visible = True

    # Select only the short layer; the analysis layer belongs to "other img"
    # and must therefore be hidden.
    plotter._update_layer_visibility_for_selection({"img"})

    assert short.visible is True
    assert long.visible is False
    assert analysis.visible is False

    # Now select the long layer; the analysis layer must become visible.
    plotter._update_layer_visibility_for_selection({"other img"})

    assert analysis.visible is True

    plotter.deleteLater()


def test_update_grid_view_multi_layer_hides_associated_layers(
    make_viewer_model,
):
    """Multi-layer selection enables grid mode and syncs analysis layers."""
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)

    a = create_image_layer_with_phasors()
    a.name = "imgA"
    viewer.add_layer(a)
    b = create_image_layer_with_phasors()
    b.name = "imgB"
    viewer.add_layer(b)
    c = create_image_layer_with_phasors()
    c.name = "imgC"
    viewer.add_layer(c)

    comp_a = viewer.add_image(
        np.zeros((5, 5)), name="Component 1 fractions: imgA"
    )
    comp_c = viewer.add_image(
        np.zeros((5, 5)), name="Component 1 fractions: imgC"
    )
    comp_c.visible = True

    viewer.grid.enabled = False

    # Select imgA and imgB (not imgC).
    plotter._update_grid_view([a, b])

    assert viewer.grid.enabled is True
    assert a.visible is True
    assert b.visible is True
    assert comp_a.visible is True
    assert c.visible is False
    assert comp_c.visible is False

    plotter.deleteLater()


def test_get_common_harmonics_empty_layers_returns_none(make_viewer_model):
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)
    assert plotter._get_common_harmonics([]) is None
    plotter.deleteLater()


def test_get_common_harmonics_skips_layer_without_harmonics(
    make_viewer_model,
):
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)

    layer1 = create_image_layer_with_phasors()
    layer1.name = "with_harmonics"
    viewer.add_layer(layer1)

    # Layer 2 with no 'harmonics' metadata key
    layer2 = create_image_layer_with_phasors()
    layer2.name = "no_harmonics"
    layer2.metadata.pop("harmonics", None)
    viewer.add_layer(layer2)

    result = plotter._get_common_harmonics([layer1, layer2])
    # Only layer1's harmonics should be considered → returns sorted set of layer1
    assert result is not None
    expected = sorted(np.atleast_1d(layer1.metadata["harmonics"]).tolist())
    assert list(result) == expected

    plotter.deleteLater()


def test_get_common_harmonics_empty_intersection_returns_none(
    make_viewer_model,
):
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)

    layer1 = create_image_layer_with_phasors(harmonic=[1, 2])
    layer1.name = "L1"
    layer2 = create_image_layer_with_phasors(harmonic=[3, 4])
    layer2.name = "L2"
    viewer.add_layer(layer1)
    viewer.add_layer(layer2)

    result = plotter._get_common_harmonics([layer1, layer2])
    assert result is None

    plotter.deleteLater()


def test_apply_mask_to_phasor_data_unsupported_layer_returns(
    make_viewer_model,
):
    """_apply_mask_to_phasor_data returns early if mask layer is empty/wrong type."""
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)

    image_layer = create_image_layer_with_phasors()
    viewer.add_layer(image_layer)

    # An Image layer is neither Shapes nor (Labels with data.any()) → early return
    other_image = create_image_layer_with_phasors()
    plotter._apply_mask_to_phasor_data(other_image, image_layer)

    # No mask metadata should have been set
    assert "mask" not in image_layer.metadata

    plotter.deleteLater()


def test_apply_mask_to_phasor_data_invalidates_features_cache(
    make_viewer_model,
):
    """_apply_mask_to_phasor_data invalidates the cache after mutating G/S."""
    from napari.layers import Labels

    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)

    image_layer = create_image_layer_with_phasors()
    viewer.add_layer(image_layer)

    mask_data = np.ones(image_layer.data.shape, dtype=np.uint8)
    mask_layer = Labels(mask_data, name="mask")
    viewer.add_layer(mask_layer)

    # Prime the features cache
    plotter.get_merged_features()
    sentinel = ('stale-mask-test',)
    plotter._features_cache = sentinel

    plotter._apply_mask_to_phasor_data(mask_layer, image_layer)

    assert plotter._features_cache is not sentinel

    plotter.deleteLater()


def test_restore_original_phasor_data_invalidates_features_cache(
    make_viewer_model,
):
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)

    image_layer = create_image_layer_with_phasors()
    viewer.add_layer(image_layer)

    plotter.get_merged_features()
    sentinel = ('stale-restore-test',)
    plotter._features_cache = sentinel

    plotter._restore_original_phasor_data(image_layer)

    assert plotter._features_cache is not sentinel

    plotter.deleteLater()


def test_invalidate_features_cache_clears_both_fields(make_viewer_model):
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)

    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)
    plotter.get_merged_features()
    assert plotter._features_cache is not None
    assert plotter._features_cache_key is not None

    plotter._invalidate_features_cache()

    assert plotter._features_cache is None
    assert plotter._features_cache_key is None

    plotter.deleteLater()


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


def test_refresh_phasor_data_no_layer_returns_early(make_viewer_model):
    """refresh_phasor_data is a no-op when no layer is selected."""
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)

    # No layers added — combobox is empty
    plotter.refresh_phasor_data()
    # Should not raise; arrays remain None
    assert plotter._g_array is None

    plotter.deleteLater()


def test_on_mask_data_changed_single_layer_mismatched_combobox_returns(
    make_viewer_model,
):
    """_on_mask_data_changed returns early in single-layer mode if mask doesn't match."""
    from napari.layers import Labels

    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)

    image_layer = create_image_layer_with_phasors()
    viewer.add_layer(image_layer)

    mask = Labels(
        np.ones(image_layer.data.shape, dtype=np.uint8),
        name="some_mask",
    )
    viewer.add_layer(mask)

    # Combobox not selecting this mask
    plotter.mask_layer_combobox.setCurrentText("None")

    # Build a fake event whose .source is the mask layer
    class _Event:
        source = mask

    g_before = image_layer.metadata['G'].copy()
    plotter._on_mask_data_changed(_Event())
    # Data should not have been modified (early return)
    np.testing.assert_array_equal(image_layer.metadata['G'], g_before)

    plotter.deleteLater()


def test_on_mask_data_changed_no_selected_layers_returns(make_viewer_model):
    """_on_mask_data_changed early-returns when no layers selected."""
    from napari.layers import Labels

    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)

    mask = Labels(np.ones((4, 4), dtype=np.uint8), name="mask")
    viewer.add_layer(mask)

    class _Event:
        source = mask

    # Should not raise; just early-returns
    plotter._on_mask_data_changed(_Event())
    plotter.deleteLater()


def test_apply_layer_data_immediate_restore_phasor_mapping(make_viewer_model):
    """_apply_layer_data calls _restore_on_layer_change when phasor_mapping_tab is current."""
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)
    layer = create_image_layer_with_phasors()
    layer.name = "L1"
    viewer.add_layer(layer)

    plotter.tab_widget.setCurrentWidget(plotter.phasor_mapping_tab)

    with patch.object(
        plotter.phasor_mapping_tab, '_restore_on_layer_change'
    ) as mock_restore:
        plotter._apply_layer_data("L1", reset_zoom=False, sync_frequency=False)
        # Should be called at least once (current tab branch)
        assert mock_restore.call_count >= 1

    plotter.deleteLater()


def test_apply_layer_data_immediate_restore_components(make_viewer_model):
    """_apply_layer_data calls _restore_on_layer_change when components_tab is current."""
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)
    layer = create_image_layer_with_phasors()
    layer.name = "L1"
    viewer.add_layer(layer)

    plotter.tab_widget.setCurrentWidget(plotter.components_tab)

    with patch.object(
        plotter.components_tab, '_restore_on_layer_change'
    ) as mock_restore:
        plotter._apply_layer_data("L1", reset_zoom=False, sync_frequency=False)
        assert mock_restore.call_count >= 1

    plotter.deleteLater()


def test_apply_layer_data_immediate_restore_fret(make_viewer_model):
    """_apply_layer_data calls _restore_on_layer_change when fret_tab is current."""
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)
    layer = create_image_layer_with_phasors()
    layer.name = "L1"
    viewer.add_layer(layer)

    plotter.tab_widget.setCurrentWidget(plotter.fret_tab)

    with patch.object(
        plotter.fret_tab, '_restore_on_layer_change'
    ) as mock_restore:
        plotter._apply_layer_data("L1", reset_zoom=False, sync_frequency=False)
        assert mock_restore.call_count >= 1

    plotter.deleteLater()


def test_apply_layer_data_marks_other_tabs_needs_update(make_viewer_model):
    """_apply_layer_data marks non-current deferrable tabs as needing update."""
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)
    layer = create_image_layer_with_phasors()
    layer.name = "L1"
    viewer.add_layer(layer)

    # Switch to a non-deferrable tab so all 3 deferrable tabs get _needs_update
    plotter.tab_widget.setCurrentWidget(plotter.filter_tab)
    plotter.phasor_mapping_tab._needs_update = False
    plotter.components_tab._needs_update = False
    plotter.fret_tab._needs_update = False

    plotter._apply_layer_data("L1", reset_zoom=False, sync_frequency=False)

    assert plotter.phasor_mapping_tab._needs_update is True
    assert plotter.components_tab._needs_update is True
    assert plotter.fret_tab._needs_update is True

    plotter.deleteLater()


def test_on_harmonic_changed_invalidates_features_cache(make_viewer_model):
    """Changing harmonic invalidates the features cache."""
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)

    plotter.get_merged_features()
    sentinel = ('stale-harmonic-test',)
    plotter._features_cache = sentinel

    plotter._on_harmonic_changed(2)

    assert plotter._features_cache is not sentinel

    plotter.deleteLater()


def test_on_mask_layer_changed_no_selected_layers_returns(make_viewer_model):
    """_on_mask_layer_changed returns early if no layers selected."""
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)

    # No image layers selected — early return
    plotter._on_mask_layer_changed("None")
    # Should not raise

    plotter.deleteLater()


def test_on_mask_data_changed_multi_layer_per_mask_filter(make_viewer_model):
    """_on_mask_data_changed multi-layer mode filters by per-layer assignment."""
    from napari.layers import Labels

    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)

    layer1 = create_image_layer_with_phasors()
    layer1.name = "img1"
    layer2 = create_image_layer_with_phasors()
    layer2.name = "img2"
    viewer.add_layer(layer1)
    viewer.add_layer(layer2)

    # Partial mask so _apply_mask_to_phasor_data actually mutates G/S
    mask_data = np.zeros(layer1.data.shape, dtype=np.uint8)
    mask_data[..., 0] = 1  # only first column is "inside" the mask
    mask = Labels(mask_data, name="mask_a")
    viewer.add_layer(mask)

    # Select both image layers (multi-layer mode)
    plotter.image_layers_checkable_combobox.setCheckedItems(["img1", "img2"])

    # Assign mask only to img1
    plotter._mask_assignments = {"img1": "mask_a"}

    class _Event:
        source = mask

    g_before_layer2 = layer2.metadata['G'].copy()

    plotter._on_mask_data_changed(_Event())

    # img2 should NOT have changed (no mask assigned to it)
    np.testing.assert_array_equal(layer2.metadata['G'], g_before_layer2)
    # img1 G should now contain NaNs from the mask application
    assert np.isnan(layer1.metadata['G']).any()

    plotter.deleteLater()


def test_on_mask_data_changed_multi_layer_no_assignments_returns(
    make_viewer_model,
):
    """_on_mask_data_changed returns early in multi-layer if no layer is assigned to this mask."""
    from napari.layers import Labels

    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)

    layer1 = create_image_layer_with_phasors()
    layer1.name = "img1"
    layer2 = create_image_layer_with_phasors()
    layer2.name = "img2"
    viewer.add_layer(layer1)
    viewer.add_layer(layer2)

    mask = Labels(
        np.ones(layer1.data.shape, dtype=np.uint8), name="some_other_mask"
    )
    viewer.add_layer(mask)

    plotter.image_layers_checkable_combobox.setCheckedItems(["img1", "img2"])

    # No assignments referencing 'some_other_mask'
    plotter._mask_assignments = {}

    class _Event:
        source = mask

    g_before_layer1 = layer1.metadata['G'].copy()
    plotter._on_mask_data_changed(_Event())
    # No change — early return because affected_layers is empty
    np.testing.assert_array_equal(layer1.metadata['G'], g_before_layer1)

    plotter.deleteLater()


def test_contour_layer_settings_clicked(make_viewer_model, monkeypatch):
    """Test _on_contour_layer_settings_clicked callback."""
    from qtpy.QtWidgets import QDialog

    from napari_phasors.plotter import PlotterWidget

    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)

    # Needs multiple layers selected
    plotter.get_selected_layer_names = lambda: ["Layer1", "Layer2"]

    class MockDialog:
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

    import napari_phasors.plotter

    monkeypatch.setattr(
        napari_phasors.plotter, "ContourLayerSettingsDialog", MockDialog
    )

    plotter._on_contour_layer_settings_clicked()

    assert plotter._contour_display_mode == "individual"
    assert plotter._contour_layer_colors == {"Layer1": (1, 0, 0, 1)}


def test_phasor_center_settings_clicked(make_viewer_model, monkeypatch):
    """Test _on_phasor_center_settings_clicked callback."""
    from qtpy.QtWidgets import QDialog

    from napari_phasors.plotter import PlotterWidget

    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)

    # Needs multiple layers selected
    plotter.get_selected_layer_names = lambda: ["Layer1", "Layer2"]

    class MockDialog:
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

    import napari_phasors.plotter

    monkeypatch.setattr(
        napari_phasors.plotter, "PhasorCenterLayerSettingsDialog", MockDialog
    )

    plotter._on_phasor_center_configure_clicked()

    assert plotter._phasor_center_display_mode == "merged"
    assert plotter._phasor_center_color == (0, 1, 0, 1)


def test_single_contour_color_clicked(make_viewer_model, monkeypatch):
    """Test _on_single_contour_color_clicked in HISTOGRAM2D and CONTOUR modes."""
    from qtpy.QtGui import QColor

    from napari_phasors.plotter import PlotterWidget

    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)

    class MockColorDialog:
        @staticmethod
        def getColor(parent=None):
            return QColor(0, 255, 0)

    import qtpy.QtWidgets

    monkeypatch.setattr(qtpy.QtWidgets, "QColorDialog", MockColorDialog)

    # Test HISTOGRAM2D
    plotter.plot_type = 'HISTOGRAM2D'
    with monkeypatch.context() as m:
        m.setattr(plotter, 'plot', lambda: None)
        m.setattr(plotter, 'refresh_current_plot', lambda: None)
        plotter._on_single_contour_color_clicked()
    assert plotter._histogram_style == 'solid'
    assert plotter._histogram_color == (0.0, 1.0, 0.0)

    # Test CONTOUR
    plotter.plot_type = 'CONTOUR'
    with monkeypatch.context() as m:
        m.setattr(plotter, 'plot', lambda: None)
        m.setattr(plotter, 'refresh_current_plot', lambda: None)
        plotter._on_single_contour_color_clicked()
    assert plotter._single_contour_style == 'solid'
    assert plotter._single_contour_color == (0.0, 1.0, 0.0)


def test_add_lifetime_ticks(make_viewer_model):
    """Test the _add_lifetime_ticks method for drawing semi-circle ticks."""
    import matplotlib.pyplot as plt

    from napari_phasors.plotter import PlotterWidget

    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)
    fig, ax = plt.subplots()
    plotter.canvas_widget = type(
        'MockCanvas', (), {'width': lambda *args, **kwargs: 300}
    )()

    plotter._get_frequency_from_layer = lambda: 80.0
    plotter._add_lifetime_ticks_to_semicircle(
        ax, visible=True, alpha=1.0, zorder=10
    )

    assert len(plotter.semi_circle_plot_artist_list) > 0
    plt.close(fig)


def test_on_colormap_changed(make_viewer_model, monkeypatch):
    from napari_phasors.plotter import PlotterWidget

    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)

    plotter.plot_type = 'CONTOUR'
    plotter.get_selected_layer_names = lambda: ["Layer1"]

    with monkeypatch.context() as m:
        m.setattr(plotter, 'plot', lambda: None)
        m.setattr(plotter, 'refresh_current_plot', lambda: None)

        # Single selected layer contour
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

        # Test other
        plotter.plot_type = 'HISTOGRAM2D'
        plotter.plotter_inputs_widget.colormap_combobox.setCurrentText(
            "plasma"
        )
        plotter._on_colormap_changed()
        assert plotter.histogram_colormap == "plasma"


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


def test_plot_colors_save_restore(make_viewer_model, qtbot, monkeypatch):
    """Saving, overriding and restoring plot colors round-trips."""
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)
    saved_colors = plotter._capture_plot_colors()
    assert isinstance(saved_colors, dict)
    assert "axes" in saved_colors
    plotter._apply_plot_colors("red")
    plotter._apply_plot_colors_from_saved(saved_colors)


def test_individual_layers_modes_and_view_helpers(
    make_viewer_model, qtbot, monkeypatch
):
    """One widget covers 'Individual layers' contour + histogram rendering,
    scroll zoom, toolbar release patches and dock helpers."""
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)

    x_data = np.array([1.0, 2.0, 5.0, 6.0])
    y_data = np.array([3.0, 4.0, 7.0, 8.0])
    layer_feature_map = {
        "Layer1": (np.array([1.0, 2.0]), np.array([3.0, 4.0])),
        "Layer2": (np.array([5.0, 6.0]), np.array([7.0, 8.0])),
    }
    monkeypatch.setattr(
        plotter, "_get_selected_layer_feature_map", lambda: layer_feature_map
    )

    # Per-layer contour rendering creates contour artists.
    plotter._contour_display_mode = "Individual layers"
    plotter._update_contour_plot(x_data, y_data)
    assert len(plotter._contour_collections) > 0

    # Per-layer histogram rendering adds artists to the axes.
    n_artists_before = len(plotter.canvas_widget.axes.collections) + len(
        plotter.canvas_widget.axes.images
    )
    plotter._histogram_display_mode = "Individual layers"
    plotter._update_histogram_plot(x_data, y_data)
    n_artists_after = len(plotter.canvas_widget.axes.collections) + len(
        plotter.canvas_widget.axes.images
    )
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


def test_color_settings_and_signal_teardown(
    make_viewer_model, qtbot, monkeypatch
):
    """One widget covers marker-color dialog, 'Select color...' histogram
    style, white-background toggle and artist-signal teardown."""
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)

    # Marker colour picked from the dialog is stored.
    mock_color = MagicMock()
    mock_color.isValid.return_value = True
    mock_color.name.return_value = "#ff0000"
    import qtpy.QtWidgets

    monkeypatch.setattr(
        qtpy.QtWidgets.QColorDialog,
        "getColor",
        lambda *args, **kwargs: mock_color,
    )
    plotter._on_marker_color_clicked()
    assert plotter._marker_color == "#ff0000"

    # The 'Select color...' sentinel switches the histogram style to solid.
    plotter.plotter_inputs_widget.colormap_combobox.setCurrentText(
        "Select color..."
    )
    plotter._on_colormap_changed()
    assert plotter._histogram_style == "solid"

    # White-background toggle restyles without error.
    plotter.on_white_background_changed()

    # Signal teardown handles non-image layers (shapes/labels) gracefully.
    viewer.add_shapes(np.random.random((2, 2)))
    viewer.add_labels(np.random.randint(0, 2, (10, 10)))
    plotter._disconnect_all_artist_signals()
