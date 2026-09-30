"""Tests for time-lapse (stack) support in the phasor plot and histogram."""

import numpy as np
import pytest
from qtpy.QtWidgets import QDialog

from napari_phasors._synthetic_generator import (
    make_intensity_layer_with_phasors,
    make_raw_flim_data,
)
from napari_phasors._timelapse import (
    CURRENT,
    POOLED,
    AnimationExportDialog,
    FrameContext,
    TimelapseControlBar,
    build_frame_statistics_rows,
    combine_frames,
    stack_axes,
)
from napari_phasors._utils import StatisticsTableWidget
from napari_phasors.plotter import PlotterWidget

N_FRAMES = 4
STACK_SHAPE = (N_FRAMES, 5, 6)


def create_stack_layer(shape=STACK_SHAPE, name="Stack"):
    """Create an intensity layer with phasors whose data has a stack axis.

    Seven time constants over 30 pixels per frame means the decay pattern
    does not repeat frame to frame, so each frame has distinct phasor
    coordinates.
    """
    raw_flim_data = make_raw_flim_data(
        shape=shape, time_constants=[0.1, 0.5, 1, 2, 3, 4, 5]
    )
    return make_intensity_layer_with_phasors(
        raw_flim_data, harmonic=[1, 2], name=name
    )


def create_flat_layer(name="Flat"):
    """Create a plain 2D intensity layer with phasors."""
    raw_flim_data = make_raw_flim_data(
        shape=(5, 6), time_constants=[0.1, 1, 2, 3, 4, 5]
    )
    return make_intensity_layer_with_phasors(
        raw_flim_data, harmonic=[1, 2], name=name
    )


def make_plotter_with_layer(viewer, layer):
    """Add *layer* to *viewer*, select it in a fresh plotter and return it."""
    viewer.add_layer(layer)
    plotter = PlotterWidget(viewer)
    plotter.image_layers_checkable_combobox.setCheckedItems([layer.name])
    plotter._process_layer_selection_change()
    return plotter


def _run_linear_projection(components_tab):
    """Configure two components and run a Linear Projection analysis."""
    components_tab.analysis_type_combo.setCurrentText("Linear Projection")
    components_tab.components[0].g_edit.setText("0.2")
    components_tab.components[0].s_edit.setText("0.1")
    components_tab._on_component_coords_changed(0)
    components_tab.components[1].g_edit.setText("0.8")
    components_tab.components[1].s_edit.setText("0.5")
    components_tab._on_component_coords_changed(1)
    components_tab._run_analysis()


def _histogram_mean(histogram):
    """Mean of the values the histogram is currently displaying."""
    return float(np.mean(histogram._raw_valid_data))


def _expected_frame_mean(histogram, frame):
    """Mean of *frame* computed straight from the un-sliced source data."""
    values = []
    for data in histogram._frame_source_datasets.values():
        array = np.asarray(data, dtype=float)[frame].ravel()
        values.append(array[np.isfinite(array)])
    return float(np.mean(np.concatenate(values)))


def _assert_frame_source_keeps_layer_shape(histogram):
    """The recorded source arrays must keep their stack axis.

    Flattening them before the frame slice is applied silently pools every
    timepoint, which is exactly the bug this guards against.
    """
    assert histogram._frame_source_datasets
    for name, data in histogram._frame_source_datasets.items():
        assert (
            np.asarray(data).shape == STACK_SHAPE
        ), f"{name} lost its stack axis: {np.asarray(data).shape}"


# ---------------------------------------------------------------------------
# FrameContext basics
# ---------------------------------------------------------------------------


def test_frame_context_selects_one_frame_of_a_stack(make_viewer_model):
    """Only non-spatial axes are stack axes. Per-frame mode masks and slices
    exactly one frame along the chosen axis, pooled mode leaves every array
    untouched, and 2D data or an empty selection offers no frame axis."""

    class _Layer:
        def __init__(self, data):
            self.data = data

    # The last two axes are spatial; anything before them is a stack axis.
    assert stack_axes(_Layer(np.zeros((5, 6)))) == []
    assert stack_axes(_Layer(np.zeros((4, 5, 6)))) == [0]
    assert stack_axes(_Layer(np.zeros((3, 4, 5, 6)))) == [0, 1]
    assert stack_axes(_Layer(None)) == []

    def context_for(layer):
        viewer = make_viewer_model()
        viewer.add_layer(layer)
        return FrameContext(viewer, lambda: [layer])

    # 2D data must not offer a frame axis, keeping the bar hidden.
    flat = create_flat_layer()
    context = context_for(flat)
    assert context.available_axes() == []
    assert context.refresh_bounds() is False
    assert context.frame_mask(flat.data.shape) is None
    assert context.state_key() == (POOLED,)

    # Pooled mode must leave every array untouched.
    context = context_for(create_stack_layer())
    data = np.arange(np.prod(STACK_SHAPE)).reshape(STACK_SHAPE)
    assert context.frame_mask(STACK_SHAPE) is None
    assert context.flat_frame_mask(STACK_SHAPE) is None
    assert np.array_equal(context.slice_array(data), data)

    # The frame mask and slice must select exactly one frame.
    context.mode = CURRENT
    context.index = 2
    assert context.available_axes() == [0]
    assert context.n_frames == N_FRAMES
    assert context.state_key() == (CURRENT, 0, 2)
    flat_mask = context.flat_frame_mask(STACK_SHAPE)
    assert flat_mask.sum() == 5 * 6
    assert np.array_equal(
        flat_mask.reshape(STACK_SHAPE)[2], np.ones((5, 6), dtype=bool)
    )
    assert np.array_equal(context.slice_array(data), data[2])
    valid = np.ones(STACK_SHAPE, dtype=bool)
    assert context.filter_valid(valid, STACK_SHAPE).sum() == 5 * 6
    assert context.filter_valid(valid.ravel(), STACK_SHAPE).sum() == 5 * 6

    # A 4D stack exposes two axes and can be stepped along either.
    context = context_for(create_stack_layer(shape=(2, 3, 5, 6)))
    assert context.available_axes() == [0, 1]
    context.mode = CURRENT
    context.axis = 1
    context.index = 2
    assert context.n_frames == 3
    mask = context.flat_frame_mask((2, 3, 5, 6)).reshape((2, 3, 5, 6))
    assert mask[:, 2].all()
    assert not mask[:, 0].any()

    # A bar built on an empty selection must simply hide itself.
    context = FrameContext(make_viewer_model(), list)
    bar = TimelapseControlBar(context)
    try:
        assert bar.isHidden() is True
        assert context.refresh_bounds() is False
    finally:
        bar.close()


# ---------------------------------------------------------------------------
# napari dims synchronisation
# ---------------------------------------------------------------------------


def test_frame_follows_the_napari_slider(make_viewer_model):
    """napari's slider (and its playback) drives the displayed frame and the
    frame drives the slider; per-frame mode plots, caches, summarises and
    selects only that frame's samples."""
    viewer = make_viewer_model()
    plotter = make_plotter_with_layer(viewer, create_stack_layer())
    try:
        bar = plotter.timelapse_bar
        context = plotter.frame_context
        layer = plotter.get_selected_layers()[0]

        # A single stack axis needs no picker. Stepping and playback belong
        # to napari's own dimension slider: duplicating them here would give
        # two sets of controls for one piece of state.
        assert bar.axis_combobox.isVisibleTo(bar) is False
        for removed in (
            "play_button",
            "frame_slider",
            "frame_label",
            "fps_spinbox",
        ):
            assert not hasattr(bar, removed), f"{removed} is back"

        pooled_g, pooled_s = plotter.get_merged_features()
        assert plotter._get_layer_phasor_samples(layer)[0].size == np.prod(
            STACK_SHAPE
        )
        pooled_map = plotter._get_selected_layer_feature_map()
        pooled_map_size = next(iter(pooled_map.values()))[0].size

        # Animations can only be exported frame by frame.
        assert bar.export_button.isEnabled() is False
        bar.mode_combobox.setCurrentIndex(bar.mode_combobox.findData(CURRENT))
        assert context.is_per_frame
        assert bar.export_button.isEnabled() is True

        # The slider moves the frame, and the frame moves the slider.
        viewer.dims.set_current_step(0, 3)
        assert context.index == 3
        context.index = 2
        assert viewer.dims.current_step[0] == 2

        # Stepping through the stack, as napari's play button does.
        for frame in range(N_FRAMES):
            viewer.dims.set_current_step(0, frame)
            assert context.index == frame
            assert plotter.get_merged_features()[0].size == 5 * 6

        # Only the displayed frame is plotted, and switching frames must
        # not serve a stale cached feature set.
        context.index = 0
        first = plotter.get_merged_features()[0].copy()
        context.index = 1
        frame_g, frame_s = plotter.get_merged_features()
        assert frame_g.size == pooled_g.size // N_FRAMES
        assert frame_s.size == pooled_s.size // N_FRAMES
        expected = layer.metadata["G"][0][1].ravel()
        expected = expected[~np.isnan(expected)]
        assert np.allclose(np.sort(frame_g), np.sort(expected))
        assert not np.array_equal(first, frame_g)
        assert plotter._features_cache_key[-1] == (CURRENT, 0, 1)

        # Contour data (one entry per layer) follows the frame too.
        per_frame_map = plotter._get_selected_layer_feature_map()
        assert next(iter(per_frame_map.values()))[0].size == (
            pooled_map_size // N_FRAMES
        )

        # Phasor-center statistics summarise only the visible frame.
        context.index = 2
        per_frame = plotter._get_layer_phasor_samples(layer)
        assert per_frame[0].size == np.prod(STACK_SHAPE) // N_FRAMES
        assert np.allclose(per_frame[0], layer.data[2].ravel())
        assert plotter._compute_single_center(layer) is not None

        # The per-point selection array matches the plotted sample count.
        selection_tab = plotter.selection_tab
        selection_tab.selection_mode_combobox.setCurrentText(
            "Manual Selection"
        )
        g = layer.metadata["G"][0]
        s = layer.metadata["S"][0]
        n_plotted = plotter.get_merged_features()[0].size
        assert selection_tab._frame_valid_mask(g, s).sum() == n_plotted

        context.mode = POOLED
        assert plotter.get_merged_features()[0].size == pooled_g.size
    finally:
        plotter.close()


# ---------------------------------------------------------------------------
# Phasor plot features
# ---------------------------------------------------------------------------


def _histogram_norm(plotter):
    """Return the ``(vmin, vmax)`` the 2D histogram is coloured with."""
    artist = plotter.canvas_widget.artists['HISTOGRAM2D']
    norm = artist._get_normalization(artist.histogram[0], is_overlay=False)
    return float(norm.vmin), float(norm.vmax)


def _histogram_grid(plotter):
    """Return a hashable description of the 2D histogram's bin grid."""
    artist = plotter.canvas_widget.artists['HISTOGRAM2D']
    _counts, x_edges, y_edges = artist.histogram
    return (
        len(x_edges),
        len(y_edges),
        float(x_edges[0]),
        float(x_edges[-1]),
        float(y_edges[0]),
        float(y_edges[-1]),
    )


def test_histogram_colour_scale_is_fixed_across_frames(make_viewer_model):
    """The 2D histogram must not rescale its colours frame by frame.

    Both the colour normalisation and the bin grid come from the whole
    acquisition (per-frame bins reuse the grid the pooled plot draws), so a
    colour means the same pixel count at every timepoint and the colorbar
    stops jumping around while stepping. Pooled plots keep biaplotter's own
    colour scale.
    """
    viewer = make_viewer_model()
    plotter = make_plotter_with_layer(viewer, create_stack_layer())
    try:
        artist = plotter.canvas_widget.artists['HISTOGRAM2D']
        assert (
            getattr(artist, '_napari_phasors_fixed_counts_range', None) is None
        )
        assert plotter._frame_histogram_reference() is None
        pooled_grid = _histogram_grid(plotter)

        plotter.frame_context.mode = CURRENT
        assert artist._napari_phasors_fixed_counts_range is not None
        viewer.dims.set_current_step(0, 1)
        assert _histogram_grid(plotter) == pooled_grid

        norms = set()
        grids = set()
        for frame in range(N_FRAMES):
            viewer.dims.set_current_step(0, frame)
            norms.add(_histogram_norm(plotter))
            grids.add(_histogram_grid(plotter))
        assert len(norms) == 1, f"colour scale changed between frames: {norms}"
        assert len(grids) == 1, f"bin grid changed between frames: {grids}"

        # The scale must span the busiest frame, not one frame's own max.
        reference = plotter._frame_histogram_reference()
        assert norms.pop() == (reference["vmin"], reference["vmax"])

        # Stepping frames must not recompute the whole-stack range, but
        # changing the bin count invalidates it through the cache key.
        viewer.dims.set_current_step(0, 2)
        assert plotter._frame_histogram_reference() is reference
        plotter.histogram_bins = plotter.histogram_bins + 10
        assert plotter._frame_histogram_reference() is not reference

        # Log colouring must be pinned across frames as well.
        plotter.plotter_inputs_widget.log_scale_checkbox.setChecked(True)
        norms = set()
        for frame in range(N_FRAMES):
            viewer.dims.set_current_step(0, frame)
            norms.add(_histogram_norm(plotter))
        assert len(norms) == 1
        vmin, _vmax = norms.pop()
        # LogNorm cannot start at a non-positive value.
        assert vmin > 0

        # Every plot type must render without error from a single frame.
        for plot_type in ("HISTOGRAM2D", "SCATTER", "CONTOUR"):
            plotter.switch_plot_type(plot_type)
            plotter.frame_context.index = 1
            plotter.plot()
        plotter.switch_plot_type("HISTOGRAM2D")
        artist = plotter.canvas_widget.artists['HISTOGRAM2D']

        plotter.frame_context.mode = POOLED
        assert artist._napari_phasors_fixed_counts_range is None

        # ``plot`` blanks the canvas when a frame yields no features at all.
        plotter.frame_context.mode = CURRENT
        plotter.plot()
        assert artist.visible is True
        plotter.get_features = lambda: None
        plotter.plot()
        assert artist.visible is False
        assert plotter._frame_plot_blanked is True
    finally:
        plotter.close()


def test_frames_across_several_selected_layers(make_viewer_model):
    """With several stacks selected the colour range covers all of them and
    the statistics table has one row per frame per layer, grouped by frame;
    a plain 2D layer beside a stack contributes to every frame."""
    viewer = make_viewer_model()
    first = create_stack_layer(name="First")
    second = create_stack_layer(name="Second")
    flat = create_flat_layer(name="Flat")
    viewer.add_layer(first)
    viewer.add_layer(second)
    viewer.add_layer(flat)

    plotter = PlotterWidget(viewer)
    combobox = plotter.image_layers_checkable_combobox
    combobox.setCheckedItems([first.name, second.name])
    plotter._process_layer_selection_change()
    try:
        _run_mapping_analysis(plotter)
        table = _mapping_stats_dock(plotter).layer_stats_table
        plotter.frame_context.mode = CURRENT

        norms = set()
        grids = set()
        for frame in range(N_FRAMES):
            viewer.dims.set_current_step(0, frame)
            norms.add(_histogram_norm(plotter))
            grids.add(_histogram_grid(plotter))
            # Both layers contribute to every frame.
            assert plotter.get_merged_features()[0].size == 2 * 5 * 6
        assert len(norms) == 1
        assert len(grids) == 1

        viewer.dims.set_current_step(0, 1)
        assert table.rowCount() == 2 * N_FRAMES
        # Rows are grouped by frame, so a frame's layers sit side by side.
        assert [row[0] for row in _table_rows(table)] == [
            str(frame) for frame in range(N_FRAMES) for _ in range(2)
        ]
        assert _highlighted_frames(table) == ["1", "1"]

        combobox.setCheckedItems([first.name, flat.name])
        plotter._process_layer_selection_change()
        plotter.frame_context.mode = CURRENT
        assert plotter._frame_histogram_reference() is not None
        # One stack frame (30 px) plus the whole 2D layer (30 px).
        assert plotter.get_merged_features()[0].size == 2 * 5 * 6
    finally:
        plotter.close()


# ---------------------------------------------------------------------------
# Control bar
# ---------------------------------------------------------------------------


def test_masked_frames_blank_the_plot(make_viewer_model):
    """A fully masked frame must not keep showing the previous frame, and a
    fully masked stack yields no usable colour range."""
    viewer = make_viewer_model()
    layer = create_stack_layer()
    # Mask out the whole second frame, as a threshold would.
    layer.metadata["G"][:, 1] = np.nan
    layer.metadata["S"][:, 1] = np.nan
    plotter = make_plotter_with_layer(viewer, layer)
    try:
        plotter.frame_context.mode = CURRENT
        plotter.frame_context.index = 0
        plotter.plot()
        assert plotter.canvas_widget.artists['HISTOGRAM2D'].visible is True

        plotter.frame_context.index = 1
        assert plotter.get_merged_features() is None
        assert plotter.canvas_widget.artists['HISTOGRAM2D'].visible is False

        plotter.frame_context.index = 2
        assert plotter.canvas_widget.artists['HISTOGRAM2D'].visible is True

        all_nan = create_stack_layer(name="All NaN")
        all_nan.metadata["G"][:] = np.nan
        all_nan.metadata["S"][:] = np.nan
        viewer.add_layer(all_nan)
        plotter.image_layers_checkable_combobox.setCheckedItems([all_nan.name])
        plotter._process_layer_selection_change()
        plotter.frame_context.mode = CURRENT
        assert plotter._frame_histogram_reference() is None
    finally:
        plotter.close()


def test_2d_data_has_no_frame_controls(
    make_viewer_model, monkeypatch, tmp_path
):
    """Plain 2D workflows never see the time-lapse controls, and without a
    stack axis the phasor-center export has no menu to show."""
    viewer = make_viewer_model()
    plotter = make_plotter_with_layer(viewer, create_flat_layer())
    try:
        # ``isHidden`` (not ``isVisible``) is the right check here: the
        # widget tree is never shown in tests, so ``isVisible`` is False
        # for every widget regardless of our own setVisible calls.
        assert plotter.timelapse_bar.isHidden() is True
        assert plotter.frame_context.available_axes() == []

        target = tmp_path / "flat.csv"
        monkeypatch.setattr(
            "napari_phasors.plotter.QMenu.exec_",
            lambda self, *a, **k: pytest.fail("menu opened for 2D data"),
            raising=False,
        )
        _accept_save_dialog(monkeypatch, target)
        plotter._export_phasor_center_statistics()
        assert len(target.read_text().strip().splitlines()) == 2
    finally:
        plotter.close()


def test_a_4d_stack_lets_the_user_pick_the_frame_axis(make_viewer_model):
    """More than one stack axis means the user gets to choose; mode and axis
    persist on the layer like every other plot setting, and switching axis
    matters only when a single frame is displayed."""
    viewer = make_viewer_model()
    plotter = make_plotter_with_layer(
        viewer, create_stack_layer(shape=(2, 3, 5, 6))
    )
    try:
        bar = plotter.timelapse_bar
        assert bar.axis_combobox.isVisibleTo(bar) is True
        assert bar.axis_combobox.count() == 2

        layer = plotter.get_selected_layers()[0]
        plotter.frame_context.axis = 1
        plotter.frame_context.mode = CURRENT
        settings = layer.metadata["settings"]
        assert settings["timelapse_mode"] == CURRENT
        assert settings["timelapse_axis"] == 1

        # Reset the in-memory state without writing it back to the layer, so
        # the restore below has something different to restore.
        plotter._updating_settings = True
        try:
            plotter.frame_context.mode = POOLED
            plotter.frame_context.axis = 0
        finally:
            plotter._updating_settings = False
        plotter._restore_plot_settings_from_metadata()
        assert plotter.frame_context.mode == CURRENT
        assert plotter.frame_context.axis == 1

        plotter.frame_context.mode = POOLED
        replots = []
        plotter._replot_for_frame_state = lambda: replots.append(True)
        plotter._on_frame_axis_changed(1)
        assert replots == []
        plotter.frame_context.mode = CURRENT
        replots.clear()
        plotter._on_frame_axis_changed(0)
        assert replots
    finally:
        plotter.close()


# ---------------------------------------------------------------------------
# Settings persistence
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Statistics and animation helpers
# ---------------------------------------------------------------------------


def test_frame_statistics_and_animation_helpers(
    make_viewer_model, qtbot, tmp_path
):
    """Statistics rows cover every frame in order (2D data is frame 0), the
    export dialog reports a normalised inclusive range, side-by-side
    figures are padded rather than scaled, and exporting nothing fails
    cleanly rather than raising."""
    from napari_phasors._timelapse import export_animation

    viewer = make_viewer_model()
    layer = create_stack_layer()
    viewer.add_layer(layer)
    context = FrameContext(viewer, lambda: [layer])
    data = np.arange(np.prod(STACK_SHAPE), dtype=float).reshape(STACK_SHAPE)
    rows = build_frame_statistics_rows({"A": data, "B": data * 2}, context)
    assert len(rows) == 2 * N_FRAMES
    assert [row["Frame"] for row in rows] == sorted(
        row["Frame"] for row in rows
    )
    first = next(
        row for row in rows if row["Frame"] == 0 and row["Name"] == "A"
    )
    assert first["Mean"] == pytest.approx(np.mean(data[0]))

    flat_viewer = make_viewer_model()
    flat = create_flat_layer()
    flat_viewer.add_layer(flat)
    rows = build_frame_statistics_rows(
        {"A": np.ones((5, 6))}, FrameContext(flat_viewer, lambda: [flat])
    )
    assert len(rows) == 1
    assert rows[0]["Frame"] == 0
    assert rows[0]["Mean"] == pytest.approx(1.0)

    dialog = AnimationExportDialog(
        n_frames=N_FRAMES, histogram_available=False, fps=8
    )
    qtbot.addWidget(dialog)
    try:
        # Histogram cannot be selected when no histogram is displayed.
        assert dialog.histogram_checkbox.isEnabled() is False
        options = dialog.get_options()
        assert options["include_phasor"] is True
        assert options["frames"] == list(range(N_FRAMES))
        assert options["fps"] == pytest.approx(8.0)
        # A reversed range is normalised rather than producing no frames.
        dialog.first_spinbox.setValue(3)
        dialog.last_spinbox.setValue(2)
        assert dialog.get_options()["frames"] == [1, 2]
    finally:
        dialog.close()

    left = np.zeros((10, 4, 3), dtype=np.uint8)
    right = np.zeros((6, 5, 3), dtype=np.uint8)
    assert combine_frames([]) is None
    assert combine_frames([left]).shape == (10, 4, 3)
    assert combine_frames([left, right]).shape == (10, 9, 3)

    assert export_animation(str(tmp_path / "empty.gif"), [], 5) is False


def test_phasor_center_export(make_viewer_model, monkeypatch, tmp_path):
    """Phasor-center export offers one pooled row or one row per frame,
    through a menu and a save dialog either of which can be dismissed, and
    warns when there are no centers to write."""
    from napari_phasors._utils import write_rows_to_csv

    viewer = make_viewer_model()
    plotter = make_plotter_with_layer(viewer, create_stack_layer())

    def fail(message):
        return lambda *a, **k: pytest.fail(message)

    def use_save_path(get_save_file_name):
        monkeypatch.setattr(
            "napari_phasors.plotter.QFileDialog.getSaveFileName",
            staticmethod(get_save_file_name),
        )

    try:
        pooled_rows = plotter._phasor_center_statistics_rows(per_frame=False)
        assert len(pooled_rows) == 1
        assert pooled_rows[0]["Frame"] == 0
        frame_rows = plotter._phasor_center_statistics_rows(per_frame=True)
        assert len(frame_rows) == N_FRAMES
        assert [row["Frame"] for row in frame_rows] == list(range(N_FRAMES))
        assert set(frame_rows[0]) == {
            "Frame",
            "Name",
            "G (center)",
            "S (center)",
            "Phase (deg)",
            "Modulation",
        }
        direct = tmp_path / "direct.csv"
        write_rows_to_csv(str(direct), frame_rows)
        lines = direct.read_text().strip().splitlines()
        assert len(lines) == N_FRAMES + 1
        assert lines[0].startswith("Frame,Name,G (center)")

        # 'Per timepoint' writes one row per frame, adding the extension.
        _choose_menu_action(monkeypatch, 1)
        _accept_save_dialog(monkeypatch, tmp_path / "centers")
        plotter._export_phasor_center_statistics()
        lines = (tmp_path / "centers.csv").read_text().strip().splitlines()
        assert len(lines) == N_FRAMES + 1
        assert lines[0].startswith("Frame,Name,G (center)")

        # 'All timepoints pooled' writes a single row.
        _choose_menu_action(monkeypatch, 0)
        _accept_save_dialog(monkeypatch, tmp_path / "pooled.csv")
        plotter._export_phasor_center_statistics()
        lines = (tmp_path / "pooled.csv").read_text().strip().splitlines()
        assert len(lines) == 2

        # Dismissing the menu, or the file dialog, writes nothing.
        _choose_menu_action(monkeypatch, None)
        use_save_path(fail("save dialog opened"))
        plotter._export_phasor_center_statistics()
        _choose_menu_action(monkeypatch, 0)
        use_save_path(lambda *a, **k: ("", ""))
        with monkeypatch.context() as patched:
            patched.setattr(
                "napari_phasors.plotter.write_rows_to_csv",
                fail("wrote without a path"),
            )
            plotter._export_phasor_center_statistics()

        # A selection with no computable centers warns instead of writing.
        warnings_seen = []
        monkeypatch.setattr(
            "napari_phasors.plotter.notifications.show_warning",
            warnings_seen.append,
        )
        monkeypatch.setattr(
            plotter, "_phasor_center_statistics_rows", lambda per_frame: []
        )
        use_save_path(fail("save dialog opened"))
        plotter._export_phasor_center_statistics()
        assert warnings_seen
    finally:
        plotter.close()


# ---------------------------------------------------------------------------
# Histogram / statistics dock
# ---------------------------------------------------------------------------


def test_mapping_histogram_and_statistics_follow_the_frame(
    make_viewer_model, tmp_path
):
    """The lifetime histogram, its statistics table and its exports
    summarise one frame at a time, highlight the displayed frame and return
    to the per-layer layout when pooled; the animation can include it."""
    viewer = make_viewer_model()
    plotter = make_plotter_with_layer(viewer, create_stack_layer())
    try:
        mapping_tab = plotter.phasor_mapping_tab
        plotter.tab_widget.setCurrentWidget(mapping_tab)
        mapping_tab.frequency_input.setText("80")
        mapping_tab.calculate_output_data()
        mapping_tab.plot_lifetime_histogram()
        histogram = mapping_tab.histogram_widget
        assert plotter._active_histogram_widget() is histogram
        pooled_size = len(histogram._raw_valid_data)
        assert histogram.has_frame_source() is True

        # Per-timepoint export writes one row per frame per dataset.
        stats_dock = _mapping_stats_dock(plotter)
        per_frame_path = tmp_path / "per_frame.csv"
        stats_dock._write_frame_statistics_to_csv(
            str(per_frame_path), "per_frame"
        )
        lines = per_frame_path.read_text().strip().splitlines()
        assert lines[0].startswith("Frame,Name,Lifetime (ns) Center of Mass")
        assert len(lines) == N_FRAMES + 1
        pooled_path = tmp_path / "pooled.csv"
        stats_dock._write_frame_statistics_to_csv(str(pooled_path), "pooled")
        pooled_lines = pooled_path.read_text().strip().splitlines()
        assert len(pooled_lines) == 2
        assert pooled_lines[1].startswith("all,")

        # The Calculate button path; pooled mode keeps the historic
        # one-row-per-layer layout.
        _run_mapping_analysis(plotter)
        table = stats_dock.layer_stats_table
        assert table.rowCount() == 1
        assert _table_column_names(table)[0] == "Name"

        # In per-frame mode the table shows one row per timepoint, and the
        # statistic columns name the quantity they summarise.
        plotter.frame_context.mode = CURRENT
        assert table.rowCount() == N_FRAMES
        assert _table_column_names(table) == ["Frame", "Name"] + [
            f"Lifetime (ns) {column}"
            for column in StatisticsTableWidget.COLUMNS[1:]
        ]
        assert [row[0] for row in _table_rows(table)] == [
            str(frame) for frame in range(N_FRAMES)
        ]
        # Each row must report that frame's own mean, not a pooled one.
        for row in _table_rows(table):
            frame = int(row[0])
            assert float(row[3]) == pytest.approx(
                _expected_frame_mean(histogram, frame), abs=5e-5
            )

        # Drive it the way the user does, through the viewer slider, so a
        # broken signal chain fails here rather than being masked by an
        # explicit refresh call.
        viewer.dims.set_current_step(0, 1)
        assert plotter.frame_context.index == 1
        assert len(histogram._raw_valid_data) == pooled_size // N_FRAMES
        assert _histogram_mean(histogram) == pytest.approx(
            _expected_frame_mean(histogram, 1)
        )
        viewer.dims.set_current_step(0, 3)
        assert _histogram_mean(histogram) == pytest.approx(
            _expected_frame_mean(histogram, 3)
        )

        # Exactly the displayed frame's row is highlighted, and it follows.
        viewer.dims.set_current_step(0, 2)
        assert _highlighted_frames(table) == ["2"]
        viewer.dims.set_current_step(0, 0)
        assert _highlighted_frames(table) == ["0"]

        # Rendering both figures stacks them side by side in each frame.
        options = {
            "include_phasor": True,
            "include_histogram": True,
            "frames": [0, 1],
            "fps": 5,
        }
        both = plotter._render_animation_frames(options, histogram)
        options["include_histogram"] = False
        phasor_only = plotter._render_animation_frames(options, histogram)
        assert len(both) == 2
        assert both[0].shape[1] > phasor_only[0].shape[1]

        # Leaving per-frame mode brings back the per-layer table.
        plotter.frame_context.mode = POOLED
        assert table.rowCount() == 1
        assert _table_column_names(table)[0] == "Name"
        assert _highlighted_frames(table) == []
    finally:
        plotter.close()


def test_components_and_fret_histograms_follow_the_frame(
    make_viewer_model, tmp_path
):
    """Component fractions and FRET efficiency are summarised one frame at
    a time, and component fractions export one row per frame.

    Regression test: the component datasets used to be flattened before the
    frame slice could be applied, so the histogram (and therefore the
    statistics table and the per-timepoint export) stayed pooled.
    """
    viewer = make_viewer_model()
    plotter = make_plotter_with_layer(viewer, create_stack_layer())
    try:
        components_tab = plotter.components_tab
        _run_linear_projection(components_tab)
        histogram = components_tab.histogram_widget
        pooled_size = len(histogram._raw_valid_data)
        assert histogram.has_frame_source() is True
        _assert_frame_source_keeps_layer_shape(histogram)

        stats_dock = plotter._statistics_stack.widget(
            plotter._components_stats_page_idx
        )
        target = tmp_path / "components_per_frame.csv"
        stats_dock._write_frame_statistics_to_csv(str(target), "per_frame")
        lines = target.read_text().strip().splitlines()
        assert len(lines) == N_FRAMES + 1
        assert [line.split(",")[0] for line in lines[1:]] == [
            str(frame) for frame in range(N_FRAMES)
        ]

        plotter.frame_context.mode = CURRENT
        viewer.dims.set_current_step(0, 1)
        assert len(histogram._raw_valid_data) == pooled_size // N_FRAMES
        assert _histogram_mean(histogram) == pytest.approx(
            _expected_frame_mean(histogram, 1)
        )
        viewer.dims.set_current_step(0, 3)
        assert _histogram_mean(histogram) == pytest.approx(
            _expected_frame_mean(histogram, 3)
        )

        # FRET efficiency, calculated over the whole stack.
        plotter.frame_context.mode = POOLED
        fret_tab = plotter.fret_tab
        fret_tab.frequency_input.setText("80")
        fret_tab.donor_line_edit.setText("4.0")
        assert fret_tab._fret_validation() is None
        fret_tab.calculate_fret_efficiency()
        histogram = fret_tab.histogram_widget
        pooled_size = len(histogram._raw_valid_data)
        assert histogram.has_frame_source() is True
        _assert_frame_source_keeps_layer_shape(histogram)

        plotter.frame_context.mode = CURRENT
        viewer.dims.set_current_step(0, 2)
        assert len(histogram._raw_valid_data) == pooled_size // N_FRAMES
        assert _histogram_mean(histogram) == pytest.approx(
            _expected_frame_mean(histogram, 2)
        )
    finally:
        plotter.close()


def _run_mapping_analysis(plotter, frequency="80"):
    """Run the phasor mapping analysis the way the Calculate button does."""
    mapping_tab = plotter.phasor_mapping_tab
    mapping_tab.frequency_input.setText(frequency)
    mapping_tab._on_calculate_lifetime_clicked()
    return mapping_tab


def _mapping_stats_dock(plotter):
    """Return the statistics dock page linked to the phasor mapping tab."""
    return plotter._statistics_stack.widget(plotter._phasor_map_stats_page_idx)


def _table_column_names(table):
    """Return the table's current header labels."""
    return [
        table.horizontalHeaderItem(index).text()
        for index in range(table.columnCount())
    ]


def _table_rows(table):
    """Return the table contents as a list of row-value lists."""
    return [
        [
            table.item(row, col).text() if table.item(row, col) else ""
            for col in range(table.columnCount())
        ]
        for row in range(table.rowCount())
    ]


def _highlighted_frames(table):
    """Return the Frame values of the rows rendered as 'current'."""
    frames = []
    for row in range(table.rowCount()):
        item = table.item(row, 0)
        if item is not None and item.font().bold():
            frames.append(item.text())
    return frames


# ---------------------------------------------------------------------------
# Selections
# ---------------------------------------------------------------------------


def test_selections_drawn_on_one_frame_apply_to_every_frame(
    make_viewer_model,
):
    """A brush stroke, the eraser or a rectangle drawn on the displayed
    frame labels (or unlabels) the matching pixels in every frame."""
    viewer = make_viewer_model()
    plotter = make_plotter_with_layer(viewer, create_stack_layer())
    try:
        selection_tab = plotter.selection_tab
        selection_tab.selection_mode_combobox.setCurrentText(
            "Manual Selection"
        )
        plotter.frame_context.mode = CURRENT
        plotter.frame_context.index = 0
        canvas = plotter.canvas_widget
        layer = plotter.get_selected_layers()[0]
        g = layer.metadata["G"][0]
        s = layer.metadata["S"][0]

        class _MouseEvent:
            def __init__(self, x, y, inaxes=None):
                self.xdata = x
                self.ydata = y
                self.inaxes = inaxes
                self.button = 1

        def dab(selector, x, y):
            """Press and release the painting tool on a single spot."""
            selector._on_press(_MouseEvent(x, y, canvas.axes))
            selector._on_release(_MouseEvent(x, y, canvas.axes))

        def current_map():
            return layer.metadata["settings"]["selections"][
                "manual_selections"
            ][selection_tab.selection_id]

        canvas.brush_size = 64
        canvas.active_selector = "BRUSH"
        brush = canvas.active_selector

        # Paint on a phasor coordinate of the displayed frame; the decay
        # patterns repeat, so later frames hold matching pixels as well.
        x, y = float(g[0, 0, 0]), float(s[0, 0, 0])
        dab(brush, x, y)
        assert brush.last_geometry is not None
        covered = brush.last_geometry.contains_points(
            np.column_stack((g.ravel(), s.ravel()))
        ).reshape(g.shape)
        assert covered.any()

        selection_map = current_map()
        assert selection_map.shape == STACK_SHAPE
        # Every pixel of the stack whose phasor falls under the stroke is
        # labelled, not just the ones on the displayed frame.
        assert np.all(selection_map[covered] > 0)
        assert covered[1:].any(), "stroke should reach later frames too"

        # The eraser takes exactly those pixels back out again
        canvas.active_selector = "ERASER"
        dab(canvas.active_selector, x, y)
        assert not current_map().any()

        # A rectangle covering the whole phasor space, drawn with
        # biaplotter's own rectangle selector, labels every frame.
        canvas.active_selector = "RECTANGLE"
        selector = canvas.active_selector
        frame_g, frame_s = plotter.get_merged_features()
        selector.data = np.column_stack((frame_g, frame_s))
        selector.on_select(_MouseEvent(-2.0, -2.0), _MouseEvent(2.0, 2.0))
        # biaplotter hands back one class value per *plotted* point.
        selection_tab.manual_selection_changed(
            np.ones(frame_g.size, dtype=np.uint32)
        )
        selection_map = current_map()
        assert selection_map.shape == STACK_SHAPE
        for frame in range(N_FRAMES):
            assert selection_map[frame].any(), f"frame {frame} not labelled"
    finally:
        plotter.close()


# ---------------------------------------------------------------------------
# Animation export
# ---------------------------------------------------------------------------


def test_rendering_animation_frames(make_viewer_model, tmp_path):
    """Rendering yields one RGB image per frame in range (out-of-range
    indices are skipped, not clamped, and the displayed frame still gets
    rendered), leaves the viewer on its starting frame, and writes a GIF."""
    iio = pytest.importorskip("imageio.v3")

    from napari_phasors._timelapse import export_animation

    viewer = make_viewer_model()
    plotter = make_plotter_with_layer(viewer, create_stack_layer())
    try:
        plotter.frame_context.mode = CURRENT
        plotter.frame_context.index = 2

        def render(frames):
            options = {
                "include_phasor": True,
                "include_histogram": False,
                "frames": frames,
                "fps": 5,
            }
            return plotter._render_animation_frames(options, histogram=None)

        frames = render(list(range(N_FRAMES)))
        assert len(frames) == N_FRAMES
        assert frames[0].ndim == 3 and frames[0].shape[2] == 3
        assert plotter.frame_context.index == 2
        assert len(render([2])) == 1
        assert len(render([-1, 0, N_FRAMES, N_FRAMES + 5])) == 1

        target = tmp_path / "animation.gif"
        assert export_animation(str(target), frames, 5) is True
        assert target.exists()
        assert len(iio.imread(str(target))) == N_FRAMES
    finally:
        plotter.close()


# ---------------------------------------------------------------------------
# Export handlers (the dialog-driven entry points)
# ---------------------------------------------------------------------------


class _DialogStub:
    """Stand in for a modal dialog, returning a fixed result and options."""

    def __init__(self, accepted, options=None):
        self._accepted = accepted
        self._options = options or {}
        self.constructed_with = None

    def __call__(self, *args, **kwargs):
        self.constructed_with = (args, kwargs)
        return self

    def exec_(self):
        return self._accepted

    def exec(self):
        return self._accepted

    def get_options(self):
        return self._options


def _accept_save_dialog(monkeypatch, path):
    """Make QFileDialog.getSaveFileName return *path* without a UI."""
    monkeypatch.setattr(
        "napari_phasors.plotter.QFileDialog.getSaveFileName",
        staticmethod(lambda *a, **k: (str(path), "")),
    )


def test_export_animation_button(make_viewer_model, monkeypatch, tmp_path):
    """The Export Animation button explains itself in pooled mode, stops at
    a cancelled dialog, an empty figure choice or no path, and otherwise
    renders and saves a GIF."""
    pytest.importorskip("imageio.v3")

    viewer = make_viewer_model()
    plotter = make_plotter_with_layer(viewer, create_stack_layer())
    one_frame = {
        "include_phasor": True,
        "include_histogram": False,
        "frames": [0],
        "fps": 5,
    }

    def fail(message):
        return lambda *a, **k: pytest.fail(message)

    def use_dialog(dialog):
        monkeypatch.setattr(
            "napari_phasors.plotter.AnimationExportDialog", dialog
        )

    def use_save_path(get_save_file_name):
        monkeypatch.setattr(
            "napari_phasors.plotter.QFileDialog.getSaveFileName",
            staticmethod(get_save_file_name),
        )

    try:
        # In pooled mode no dialog is ever constructed.
        messages = []
        monkeypatch.setattr(
            "napari_phasors.plotter.notifications.show_info", messages.append
        )
        use_dialog(fail("dialog opened in pooled mode"))
        plotter._on_export_animation_clicked()
        assert messages and "Current timepoint" in messages[0]

        plotter.frame_context.mode = CURRENT

        # Rejecting the options dialog exports nothing.
        use_dialog(_DialogStub(QDialog.Rejected))
        use_save_path(fail("save dialog opened after cancel"))
        plotter._on_export_animation_clicked()

        # Deselecting both figures warns rather than writing an empty GIF.
        warnings_seen = []
        monkeypatch.setattr(
            "napari_phasors.plotter.notifications.show_warning",
            warnings_seen.append,
        )
        use_dialog(
            _DialogStub(
                QDialog.Accepted, {**one_frame, "include_phasor": False}
            )
        )
        plotter._on_export_animation_clicked()
        assert warnings_seen

        # Dismissing the file dialog exports nothing.
        use_dialog(_DialogStub(QDialog.Accepted, one_frame))
        use_save_path(lambda *a, **k: ("", ""))
        with monkeypatch.context() as patched:
            patched.setattr(
                "napari_phasors.plotter.export_animation",
                fail("exported without a path"),
            )
            plotter._on_export_animation_clicked()

        # Otherwise it renders and saves; without an extension, .gif is
        # added.
        use_dialog(
            _DialogStub(
                QDialog.Accepted,
                {**one_frame, "frames": list(range(N_FRAMES))},
            )
        )
        _accept_save_dialog(monkeypatch, tmp_path / "movie")
        plotter._on_export_animation_clicked()
        assert (tmp_path / "movie.gif").exists()
    finally:
        plotter.close()


def _choose_menu_action(monkeypatch, index):
    """Pick the *index*-th action of the next QMenu shown, or None to cancel."""

    def fake_exec(self, *args, **kwargs):
        actions = self.actions()
        return None if index is None else actions[index]

    monkeypatch.setattr(
        "napari_phasors.plotter.QMenu.exec_", fake_exec, raising=False
    )


# ---------------------------------------------------------------------------
# Guard branches
# ---------------------------------------------------------------------------


def test_frame_guard_branches(make_viewer_model):
    """Frame callbacks are inert while pooled or closing, the refresh
    helpers tolerate missing widgets, a deferred tab update skips a removed
    layer, and no usable phasors or no selection means no shared range."""
    viewer = make_viewer_model()
    layer = create_stack_layer()
    plotter = make_plotter_with_layer(viewer, layer)
    try:
        replots = []
        plotter._replot_for_frame_state = lambda: replots.append(True)

        # A frame change while pooled changes nothing on screen.
        plotter._on_frame_changed(2)
        assert replots == []

        # Queued frame callbacks must not touch a widget tearing down.
        plotter.frame_context.mode = CURRENT
        replots.clear()
        plotter._is_closing = True
        plotter._on_frame_changed(1)
        plotter._on_frame_mode_changed(POOLED)
        plotter._on_frame_axis_changed(0)
        assert replots == []
        plotter._is_closing = False
        del plotter._replot_for_frame_state

        # The refresh helpers tolerate a missing bar, and tabs that are
        # absent or lack the ``refresh_for_frame_change`` hook.
        bar = plotter.timelapse_bar
        del plotter.timelapse_bar
        plotter._refresh_timelapse_controls()
        plotter.timelapse_bar = bar
        original_tabs = (
            plotter.phasor_mapping_tab,
            plotter.components_tab,
            plotter.fret_tab,
        )
        plotter.phasor_mapping_tab = None
        plotter.components_tab = object()
        plotter._refresh_frame_dependent_tabs()
        (
            plotter.phasor_mapping_tab,
            plotter.components_tab,
            plotter.fret_tab,
        ) = original_tabs

        # No analysis run means no histogram to offer the animation export.
        plotter.tab_widget.setCurrentWidget(plotter.settings_tab)
        assert plotter._active_histogram_widget() is None

        # A pending tab update must not look up a layer that is already
        # gone: the tab-change event can arrive after the layer was removed,
        # and the restore paths index the viewer by name.
        mapping_tab = plotter.phasor_mapping_tab
        mapping_tab._needs_update = True
        restores = []
        mapping_tab._restore_on_layer_change = lambda: restores.append(True)
        plotter.get_primary_layer_name = lambda: "Gone Intensity [Phasor]"
        plotter._run_deferred_tab_update(mapping_tab)
        assert restores == []
        # Tearing down short-circuits the same way.
        plotter.get_primary_layer_name = lambda: layer.name
        plotter._is_closing = True
        plotter._run_deferred_tab_update(mapping_tab)
        assert restores == []
        plotter._is_closing = False
        # With a live layer the deferred update still runs.
        plotter._run_deferred_tab_update(mapping_tab)
        assert restores == [True]
        del mapping_tab._restore_on_layer_change
        del plotter.get_primary_layer_name

        # No selected layers means no shared range to compute.
        plotter.image_layers_checkable_combobox.setCheckedItems([])
        assert plotter._frame_histogram_reference() is None
        plotter.image_layers_checkable_combobox.setCheckedItems([layer.name])

        # A layer with no G/S contributes nothing rather than raising.
        # The cached frame range is keyed on the G/S arrays, so it is not
        # served once G is gone, without anyone invalidating it.
        plotter.frame_context.mode = CURRENT
        assert plotter._frame_histogram_reference() is not None
        plotter.frame_context.mode = POOLED
        layer.metadata.pop("G")
        assert plotter._get_layer_phasor_arrays(layer) is None
        assert list(plotter._iter_layer_gs_arrays()) == []
        assert plotter._phasor_center_statistics_rows(per_frame=True) == []
        plotter.frame_context.mode = CURRENT
        assert plotter._frame_histogram_reference() is None
    finally:
        plotter._is_closing = False
        plotter.close()


def _mask_stack_layer(viewer, layer):
    """Mask *layer* with a two-label mask covering the whole stack."""
    mask = np.zeros(layer.data.shape, dtype=int)
    mask[..., :3] = 1
    mask[..., 3:] = 2
    viewer.add_labels(mask, name="mask")
    layer.metadata['mask'] = mask
    layer.metadata['mask_invert'] = False
    return mask


def test_mask_label_split_follows_the_displayed_frame(make_viewer_model):
    """Per-label curves and per-timepoint rows are sliced with the same frame
    as the data, so each label keeps only that frame's pixels."""
    viewer = make_viewer_model()
    layer = create_stack_layer()
    mask = _mask_stack_layer(viewer, layer)
    plotter = make_plotter_with_layer(viewer, layer)
    try:
        mapping_tab = _run_mapping_analysis(plotter)
        histogram = mapping_tab.histogram_widget
        table = _mapping_stats_dock(plotter).layer_stats_table
        histogram.split_by_mask_labels = True

        assert histogram.mask_label_split_active()
        pooled = {
            name: len(values) for name, values in histogram._datasets.items()
        }
        assert len(pooled) == 2

        # In per-frame mode each label keeps only that frame's pixels: the
        # whole-layer mask has to be sliced the same way as the data.
        plotter.frame_context.mode = CURRENT
        mapping_tab.refresh_for_frame_change()
        per_frame = {
            name: len(values) for name, values in histogram._datasets.items()
        }
        assert set(per_frame) == set(pooled)
        for name, count in per_frame.items():
            assert count == pytest.approx(pooled[name] / N_FRAMES, rel=0.5)
            assert count > 0
        assert sum(per_frame.values()) <= int((mask[0] > 0).sum())

        # Per-timepoint rows are per label once the labels are separated.
        histogram.split_by_mask_labels = False
        assert table.rowCount() == N_FRAMES
        histogram.split_by_mask_labels = True
        # The un-sliced source arrays keep the stack axis so they can still
        # be sliced frame by frame after the split.
        for data in histogram.frame_source_datasets().values():
            assert np.asarray(data).shape == STACK_SHAPE
        assert table.rowCount() == 2 * N_FRAMES
        names = {row[1] for row in _table_rows(table)}
        assert len(names) == 2
        assert all("label" in name for name in names)
    finally:
        plotter.close()
