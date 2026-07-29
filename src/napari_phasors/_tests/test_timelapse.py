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


def test_stack_axes_only_counts_non_spatial_axes():
    """The last two axes are spatial; anything before them is a stack axis."""

    class _Layer:
        def __init__(self, data):
            self.data = data

    assert stack_axes(_Layer(np.zeros((5, 6)))) == []
    assert stack_axes(_Layer(np.zeros((4, 5, 6)))) == [0]
    assert stack_axes(_Layer(np.zeros((3, 4, 5, 6)))) == [0, 1]
    assert stack_axes(_Layer(None)) == []


def test_frame_context_reports_no_axes_for_2d_data(make_viewer_model):
    """2D data must not offer a frame axis, keeping the bar hidden."""
    viewer = make_viewer_model()
    layer = create_flat_layer()
    viewer.add_layer(layer)

    context = FrameContext(viewer, lambda: [layer])

    assert context.available_axes() == []
    assert context.refresh_bounds() is False
    assert context.frame_mask(layer.data.shape) is None
    assert context.state_key() == (POOLED,)


def test_frame_context_masks_and_slices_the_current_frame(make_viewer_model):
    """The frame mask and slice must select exactly one frame."""
    viewer = make_viewer_model()
    layer = create_stack_layer()
    viewer.add_layer(layer)

    context = FrameContext(viewer, lambda: [layer])
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

    data = np.arange(np.prod(STACK_SHAPE)).reshape(STACK_SHAPE)
    assert np.array_equal(context.slice_array(data), data[2])

    valid = np.ones(STACK_SHAPE, dtype=bool)
    assert context.filter_valid(valid, STACK_SHAPE).sum() == 5 * 6
    assert context.filter_valid(valid.ravel(), STACK_SHAPE).sum() == 5 * 6


def test_frame_context_pooled_mode_is_a_no_op(make_viewer_model):
    """Pooled mode must leave every array untouched."""
    viewer = make_viewer_model()
    layer = create_stack_layer()
    viewer.add_layer(layer)

    context = FrameContext(viewer, lambda: [layer])
    data = np.arange(np.prod(STACK_SHAPE)).reshape(STACK_SHAPE)

    assert context.frame_mask(STACK_SHAPE) is None
    assert context.flat_frame_mask(STACK_SHAPE) is None
    assert np.array_equal(context.slice_array(data), data)


def test_frame_context_second_axis_of_a_4d_stack(make_viewer_model):
    """A 4D stack exposes two axes and can be stepped along either."""
    viewer = make_viewer_model()
    layer = create_stack_layer(shape=(2, 3, 5, 6))
    viewer.add_layer(layer)

    context = FrameContext(viewer, lambda: [layer])
    assert context.available_axes() == [0, 1]

    context.mode = CURRENT
    context.axis = 1
    context.index = 2
    assert context.n_frames == 3

    mask = context.flat_frame_mask((2, 3, 5, 6)).reshape((2, 3, 5, 6))
    assert mask[:, 2].all()
    assert not mask[:, 0].any()


# ---------------------------------------------------------------------------
# napari dims synchronisation
# ---------------------------------------------------------------------------


def test_dims_slider_drives_the_frame(make_viewer_model):
    """Moving napari's slider must move the plotter's frame."""
    viewer = make_viewer_model()
    plotter = make_plotter_with_layer(viewer, create_stack_layer())
    try:
        plotter.frame_context.mode = CURRENT
        viewer.dims.set_current_step(0, 3)
        assert plotter.frame_context.index == 3
    finally:
        plotter.close()


def test_frame_change_drives_the_dims_slider(make_viewer_model):
    """Setting the frame must move napari's slider."""
    viewer = make_viewer_model()
    plotter = make_plotter_with_layer(viewer, create_stack_layer())
    try:
        plotter.frame_context.mode = CURRENT
        plotter.frame_context.index = 2
        assert viewer.dims.current_step[0] == 2
    finally:
        plotter.close()


# ---------------------------------------------------------------------------
# Phasor plot features
# ---------------------------------------------------------------------------


def test_merged_features_are_restricted_to_the_current_frame(
    make_viewer_model,
):
    """Per-frame mode must plot only one frame's worth of samples."""
    viewer = make_viewer_model()
    plotter = make_plotter_with_layer(viewer, create_stack_layer())
    try:
        pooled_g, pooled_s = plotter.get_merged_features()

        plotter.frame_context.mode = CURRENT
        plotter.frame_context.index = 1
        frame_g, frame_s = plotter.get_merged_features()

        assert frame_g.size == pooled_g.size // N_FRAMES
        assert frame_s.size == pooled_s.size // N_FRAMES

        layer = plotter.get_selected_layers()[0]
        harmonic_g = layer.metadata["G"][0]
        expected = harmonic_g[1].ravel()
        expected = expected[~np.isnan(expected)]
        assert np.allclose(np.sort(frame_g), np.sort(expected))

        plotter.frame_context.mode = POOLED
        assert plotter.get_merged_features()[0].size == pooled_g.size
    finally:
        plotter.close()


def test_features_cache_is_keyed_on_the_frame(make_viewer_model):
    """Switching frames must not serve a stale cached feature set."""
    viewer = make_viewer_model()
    plotter = make_plotter_with_layer(viewer, create_stack_layer())
    try:
        plotter.frame_context.mode = CURRENT
        plotter.frame_context.index = 0
        first = plotter.get_merged_features()[0].copy()

        plotter.frame_context.index = 3
        second = plotter.get_merged_features()[0]

        assert not np.array_equal(first, second)
        assert plotter._features_cache_key[-1] == (CURRENT, 0, 3)
    finally:
        plotter.close()


def test_phasor_center_samples_follow_the_frame(make_viewer_model):
    """Phasor-center statistics must summarise only the visible frame."""
    viewer = make_viewer_model()
    plotter = make_plotter_with_layer(viewer, create_stack_layer())
    try:
        layer = plotter.get_selected_layers()[0]
        pooled = plotter._get_layer_phasor_samples(layer)
        assert pooled[0].size == np.prod(STACK_SHAPE)

        plotter.frame_context.mode = CURRENT
        plotter.frame_context.index = 2
        per_frame = plotter._get_layer_phasor_samples(layer)

        assert per_frame[0].size == np.prod(STACK_SHAPE) // N_FRAMES
        assert np.allclose(per_frame[0], layer.data[2].ravel())
        assert plotter._compute_single_center(layer) is not None
    finally:
        plotter.close()


def test_per_layer_feature_map_follows_the_frame(make_viewer_model):
    """Contour data (one entry per layer) must follow the frame too."""
    viewer = make_viewer_model()
    plotter = make_plotter_with_layer(viewer, create_stack_layer())
    try:
        pooled = plotter._get_selected_layer_feature_map()
        pooled_size = next(iter(pooled.values()))[0].size

        plotter.frame_context.mode = CURRENT
        per_frame = plotter._get_selected_layer_feature_map()
        assert next(iter(per_frame.values()))[0].size == (
            pooled_size // N_FRAMES
        )
    finally:
        plotter.close()


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
    acquisition, so a colour means the same pixel count at every timepoint
    and the colorbar stops jumping around while stepping.
    """
    viewer = make_viewer_model()
    plotter = make_plotter_with_layer(viewer, create_stack_layer())
    try:
        plotter.frame_context.mode = CURRENT

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
    finally:
        plotter.close()


def test_histogram_grid_matches_pooled_mode(make_viewer_model):
    """Per-frame bins reuse the grid the pooled plot would have drawn."""
    viewer = make_viewer_model()
    plotter = make_plotter_with_layer(viewer, create_stack_layer())
    try:
        pooled_grid = _histogram_grid(plotter)

        plotter.frame_context.mode = CURRENT
        viewer.dims.set_current_step(0, 1)

        assert _histogram_grid(plotter) == pooled_grid
    finally:
        plotter.close()


def test_histogram_colour_scale_fixed_with_log_scale(make_viewer_model):
    """Log colouring must be pinned across frames as well."""
    viewer = make_viewer_model()
    plotter = make_plotter_with_layer(viewer, create_stack_layer())
    try:
        plotter.plotter_inputs_widget.log_scale_checkbox.setChecked(True)
        plotter.frame_context.mode = CURRENT

        norms = set()
        for frame in range(N_FRAMES):
            viewer.dims.set_current_step(0, frame)
            norms.add(_histogram_norm(plotter))

        assert len(norms) == 1
        vmin, _vmax = norms.pop()
        # LogNorm cannot start at a non-positive value.
        assert vmin > 0
    finally:
        plotter.close()


def test_histogram_colour_scale_spans_every_selected_layer(make_viewer_model):
    """With several stacks selected the range covers all of them."""
    viewer = make_viewer_model()
    first = create_stack_layer(name="First")
    second = create_stack_layer(name="Second")
    viewer.add_layer(first)
    viewer.add_layer(second)

    plotter = PlotterWidget(viewer)
    plotter.image_layers_checkable_combobox.setCheckedItems(
        [first.name, second.name]
    )
    plotter._process_layer_selection_change()
    try:
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
    finally:
        plotter.close()


def test_pooled_mode_keeps_biaplotter_colour_scale(make_viewer_model):
    """Pooled plots are untouched: no fixed range is imposed on them."""
    viewer = make_viewer_model()
    plotter = make_plotter_with_layer(viewer, create_stack_layer())
    try:
        artist = plotter.canvas_widget.artists['HISTOGRAM2D']
        assert (
            getattr(artist, '_napari_phasors_fixed_counts_range', None) is None
        )
        assert plotter._frame_histogram_reference() is None

        plotter.frame_context.mode = CURRENT
        assert artist._napari_phasors_fixed_counts_range is not None

        plotter.frame_context.mode = POOLED
        assert artist._napari_phasors_fixed_counts_range is None
    finally:
        plotter.close()


def test_frame_histogram_reference_is_cached(make_viewer_model):
    """Stepping frames must not recompute the whole-stack range each time."""
    viewer = make_viewer_model()
    plotter = make_plotter_with_layer(viewer, create_stack_layer())
    try:
        plotter.frame_context.mode = CURRENT
        reference = plotter._frame_histogram_reference()

        viewer.dims.set_current_step(0, 2)
        assert plotter._frame_histogram_reference() is reference

        # Changing the bin count invalidates it through the cache key.
        plotter.histogram_bins = plotter.histogram_bins + 10
        assert plotter._frame_histogram_reference() is not reference
    finally:
        plotter.close()


def test_plot_runs_in_per_frame_mode(make_viewer_model):
    """Every plot type must render without error from a single frame."""
    viewer = make_viewer_model()
    plotter = make_plotter_with_layer(viewer, create_stack_layer())
    try:
        plotter.frame_context.mode = CURRENT
        for plot_type in ("HISTOGRAM2D", "SCATTER", "CONTOUR"):
            plotter.switch_plot_type(plot_type)
            plotter.frame_context.index = 1
            plotter.plot()
    finally:
        plotter.close()


# ---------------------------------------------------------------------------
# Control bar
# ---------------------------------------------------------------------------


def test_empty_frame_blanks_the_plot(make_viewer_model):
    """A fully masked frame must not keep showing the previous frame."""
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
    finally:
        plotter.close()


def test_control_bar_is_hidden_for_2d_data(make_viewer_model):
    """Plain 2D workflows must not see the time-lapse controls at all."""
    viewer = make_viewer_model()
    plotter = make_plotter_with_layer(viewer, create_flat_layer())
    try:
        # ``isHidden`` (not ``isVisible``) is the right check here: the
        # widget tree is never shown in tests, so ``isVisible`` is False
        # for every widget regardless of our own setVisible calls.
        assert plotter.timelapse_bar.isHidden() is True
        assert plotter.frame_context.available_axes() == []
    finally:
        plotter.close()


def test_control_bar_configures_itself_for_a_stack(make_viewer_model):
    """The bar offers the mode and export, and hides the axis picker for 3D."""
    viewer = make_viewer_model()
    plotter = make_plotter_with_layer(viewer, create_stack_layer())
    try:
        bar = plotter.timelapse_bar
        # A single stack axis needs no picker.
        assert bar.axis_combobox.isVisibleTo(bar) is False

        # Animations can only be exported frame by frame.
        assert bar.export_button.isEnabled() is False
        bar.mode_combobox.setCurrentIndex(bar.mode_combobox.findData(CURRENT))
        assert plotter.frame_context.is_per_frame
        assert bar.export_button.isEnabled() is True
    finally:
        plotter.close()


def test_control_bar_has_no_playback_controls(make_viewer_model):
    """Stepping and playback belong to napari's own dimension slider.

    Duplicating them here would give two sets of controls for one piece of
    state, so the bar deliberately exposes none.
    """
    viewer = make_viewer_model()
    plotter = make_plotter_with_layer(viewer, create_stack_layer())
    try:
        bar = plotter.timelapse_bar
        for removed in (
            "play_button",
            "frame_slider",
            "frame_label",
            "fps_spinbox",
        ):
            assert not hasattr(bar, removed), f"{removed} is back"

        # The napari slider remains the way to change frames.
        plotter.frame_context.mode = CURRENT
        viewer.dims.set_current_step(0, 2)
        assert plotter.frame_context.index == 2
    finally:
        plotter.close()


def test_control_bar_axis_picker_shown_for_4d(make_viewer_model):
    """More than one stack axis means the user gets to choose."""
    viewer = make_viewer_model()
    plotter = make_plotter_with_layer(
        viewer, create_stack_layer(shape=(2, 3, 5, 6))
    )
    try:
        bar = plotter.timelapse_bar
        assert bar.axis_combobox.isVisibleTo(bar) is True
        assert bar.axis_combobox.count() == 2
    finally:
        plotter.close()


def test_napari_playback_drives_every_frame(make_viewer_model):
    """Stepping the viewer through the stack keeps the plot in step.

    This is what napari's own play button does, one step at a time.
    """
    viewer = make_viewer_model()
    plotter = make_plotter_with_layer(viewer, create_stack_layer())
    try:
        plotter.frame_context.mode = CURRENT
        for frame in range(N_FRAMES):
            viewer.dims.set_current_step(0, frame)
            assert plotter.frame_context.index == frame
            assert plotter.get_merged_features()[0].size == 5 * 6
    finally:
        plotter.close()


def test_control_bar_survives_a_context_without_layers(make_viewer_model):
    """A bar built on an empty selection must simply hide itself."""
    viewer = make_viewer_model()
    context = FrameContext(viewer, list)
    bar = TimelapseControlBar(context)
    try:
        assert bar.isHidden() is True
        assert context.refresh_bounds() is False
    finally:
        bar.close()


# ---------------------------------------------------------------------------
# Settings persistence
# ---------------------------------------------------------------------------


def test_frame_settings_round_trip_through_layer_metadata(make_viewer_model):
    """Mode and axis persist on the layer like every other plot setting."""
    viewer = make_viewer_model()
    plotter = make_plotter_with_layer(
        viewer, create_stack_layer(shape=(2, 3, 5, 6))
    )
    try:
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
    finally:
        plotter.close()


# ---------------------------------------------------------------------------
# Statistics and animation helpers
# ---------------------------------------------------------------------------


def test_build_frame_statistics_rows_covers_every_frame(make_viewer_model):
    """One row per frame and dataset, in frame order."""
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


def test_build_frame_statistics_rows_handles_2d_data(make_viewer_model):
    """2D datasets collapse to a single frame-0 row."""
    viewer = make_viewer_model()
    layer = create_flat_layer()
    viewer.add_layer(layer)
    context = FrameContext(viewer, lambda: [layer])

    rows = build_frame_statistics_rows({"A": np.ones((5, 6))}, context)
    assert len(rows) == 1
    assert rows[0]["Frame"] == 0
    assert rows[0]["Mean"] == pytest.approx(1.0)


def test_phasor_center_rows_pooled_and_per_frame(make_viewer_model):
    """Phasor-center export offers one pooled row or one row per frame."""
    viewer = make_viewer_model()
    plotter = make_plotter_with_layer(viewer, create_stack_layer())
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
    finally:
        plotter.close()


def test_phasor_center_export_writes_csv(make_viewer_model, tmp_path, qtbot):
    """The phasor-center CSV has a header plus one row per frame."""
    from napari_phasors._utils import write_rows_to_csv

    viewer = make_viewer_model()
    plotter = make_plotter_with_layer(viewer, create_stack_layer())
    try:
        rows = plotter._phasor_center_statistics_rows(per_frame=True)
        target = tmp_path / "centers.csv"
        write_rows_to_csv(str(target), rows)

        lines = target.read_text().strip().splitlines()
        assert len(lines) == N_FRAMES + 1
        assert lines[0].startswith("Frame,Name,G (center)")
    finally:
        plotter.close()


# ---------------------------------------------------------------------------
# Histogram / statistics dock
# ---------------------------------------------------------------------------


def test_histogram_follows_the_frame(make_viewer_model):
    """The 1-D histogram and its statistics summarise one frame at a time."""
    viewer = make_viewer_model()
    plotter = make_plotter_with_layer(viewer, create_stack_layer())
    try:
        mapping_tab = plotter.phasor_mapping_tab
        mapping_tab.frequency_input.setText("80")
        mapping_tab.calculate_output_data()
        mapping_tab.plot_lifetime_histogram()

        histogram = mapping_tab.histogram_widget
        pooled_size = len(histogram._raw_valid_data)
        assert histogram.has_frame_source() is True

        # Drive it the way the user does — through the viewer slider — so a
        # broken signal chain fails here rather than being masked by an
        # explicit refresh call.
        plotter.frame_context.mode = CURRENT
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
    finally:
        plotter.close()


def test_components_histogram_follows_the_frame(make_viewer_model):
    """Component fractions must be summarised one frame at a time.

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
    finally:
        plotter.close()


def test_fret_histogram_follows_the_frame(make_viewer_model):
    """FRET efficiency must be summarised one frame at a time."""
    viewer = make_viewer_model()
    plotter = make_plotter_with_layer(viewer, create_stack_layer())
    try:
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


def test_components_export_per_timepoint(make_viewer_model, tmp_path):
    """Per-timepoint export of component fractions has one row per frame."""
    viewer = make_viewer_model()
    plotter = make_plotter_with_layer(viewer, create_stack_layer())
    try:
        components_tab = plotter.components_tab
        _run_linear_projection(components_tab)

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


def test_statistics_table_lists_every_frame(make_viewer_model):
    """In per-frame mode the table shows one row per timepoint."""
    viewer = make_viewer_model()
    plotter = make_plotter_with_layer(viewer, create_stack_layer())
    try:
        _run_mapping_analysis(plotter)
        table = _mapping_stats_dock(plotter).layer_stats_table

        # Pooled mode keeps the historic one-row-per-layer layout.
        assert table.rowCount() == 1
        assert _table_column_names(table)[0] == "Name"

        plotter.frame_context.mode = CURRENT

        assert table.rowCount() == N_FRAMES
        assert _table_column_names(table) == [
            "Frame",
            "Name",
            "Center of Mass",
            "Mean",
            "Median",
            "Std Dev",
        ]
        assert [row[0] for row in _table_rows(table)] == [
            str(frame) for frame in range(N_FRAMES)
        ]

        # Each row must report that frame's own mean, not a pooled one.
        histogram = plotter.phasor_mapping_tab.histogram_widget
        for row in _table_rows(table):
            frame = int(row[0])
            assert float(row[3]) == pytest.approx(
                _expected_frame_mean(histogram, frame), abs=5e-5
            )
    finally:
        plotter.close()


def test_statistics_table_highlights_the_current_frame(make_viewer_model):
    """Exactly the displayed frame's row is highlighted, and it follows."""
    viewer = make_viewer_model()
    plotter = make_plotter_with_layer(viewer, create_stack_layer())
    try:
        _run_mapping_analysis(plotter)
        table = _mapping_stats_dock(plotter).layer_stats_table

        plotter.frame_context.mode = CURRENT
        viewer.dims.set_current_step(0, 2)
        assert _highlighted_frames(table) == ["2"]

        viewer.dims.set_current_step(0, 0)
        assert _highlighted_frames(table) == ["0"]
    finally:
        plotter.close()


def test_statistics_table_restores_pooled_layout(make_viewer_model):
    """Leaving per-frame mode brings back the per-layer table."""
    viewer = make_viewer_model()
    plotter = make_plotter_with_layer(viewer, create_stack_layer())
    try:
        _run_mapping_analysis(plotter)
        table = _mapping_stats_dock(plotter).layer_stats_table

        plotter.frame_context.mode = CURRENT
        assert table.rowCount() == N_FRAMES

        plotter.frame_context.mode = POOLED
        assert table.rowCount() == 1
        assert _table_column_names(table)[0] == "Name"
        assert _highlighted_frames(table) == []
    finally:
        plotter.close()


def test_statistics_table_frame_rows_for_multiple_layers(make_viewer_model):
    """With several layers the table has one row per frame per layer."""
    viewer = make_viewer_model()
    first = create_stack_layer(name="First")
    second = create_stack_layer(name="Second")
    viewer.add_layer(first)
    viewer.add_layer(second)

    plotter = PlotterWidget(viewer)
    plotter.image_layers_checkable_combobox.setCheckedItems(
        [first.name, second.name]
    )
    plotter._process_layer_selection_change()
    try:
        _run_mapping_analysis(plotter)
        table = _mapping_stats_dock(plotter).layer_stats_table

        plotter.frame_context.mode = CURRENT
        viewer.dims.set_current_step(0, 1)

        assert table.rowCount() == 2 * N_FRAMES
        # Rows are grouped by frame, so a frame's layers sit side by side.
        assert [row[0] for row in _table_rows(table)] == [
            str(frame) for frame in range(N_FRAMES) for _ in range(2)
        ]
        assert _highlighted_frames(table) == ["1", "1"]
    finally:
        plotter.close()


def test_statistics_dock_exports_per_timepoint(make_viewer_model, tmp_path):
    """Per-timepoint export writes one row per frame per dataset."""
    viewer = make_viewer_model()
    plotter = make_plotter_with_layer(viewer, create_stack_layer())
    try:
        mapping_tab = plotter.phasor_mapping_tab
        mapping_tab.frequency_input.setText("80")
        mapping_tab.calculate_output_data()
        mapping_tab.plot_lifetime_histogram()

        stats_dock = plotter._statistics_stack.widget(
            plotter._phasor_map_stats_page_idx
        )

        per_frame_path = tmp_path / "per_frame.csv"
        stats_dock._write_frame_statistics_to_csv(
            str(per_frame_path), "per_frame"
        )
        lines = per_frame_path.read_text().strip().splitlines()
        assert lines[0].startswith("Frame,Name,Center of Mass")
        assert len(lines) == N_FRAMES + 1

        pooled_path = tmp_path / "pooled.csv"
        stats_dock._write_frame_statistics_to_csv(str(pooled_path), "pooled")
        pooled_lines = pooled_path.read_text().strip().splitlines()
        assert len(pooled_lines) == 2
        assert pooled_lines[1].startswith("all,")
    finally:
        plotter.close()


# ---------------------------------------------------------------------------
# Selections
# ---------------------------------------------------------------------------


def test_selection_data_stays_aligned_with_the_plot(make_viewer_model):
    """The per-point selection array must match the plotted sample count."""
    viewer = make_viewer_model()
    plotter = make_plotter_with_layer(viewer, create_stack_layer())
    try:
        selection_tab = plotter.selection_tab
        selection_tab.selection_mode_combobox.setCurrentText(
            "Manual Selection"
        )

        plotter.frame_context.mode = CURRENT
        plotter.frame_context.index = 2

        layer = plotter.get_selected_layers()[0]
        g = layer.metadata["G"][0]
        s = layer.metadata["S"][0]

        n_plotted = plotter.get_merged_features()[0].size
        assert selection_tab._frame_valid_mask(g, s).sum() == n_plotted
    finally:
        plotter.close()


def test_manual_selection_applies_to_every_frame(make_viewer_model):
    """A region drawn on one frame labels matching pixels in all frames."""
    viewer = make_viewer_model()
    plotter = make_plotter_with_layer(viewer, create_stack_layer())
    try:
        selection_tab = plotter.selection_tab
        selection_tab.selection_mode_combobox.setCurrentText(
            "Manual Selection"
        )

        plotter.frame_context.mode = CURRENT
        plotter.frame_context.index = 0

        # Draw a rectangle covering the whole phasor space so every pixel
        # falls inside it, using biaplotter's own rectangle selector.
        canvas = plotter.canvas_widget
        canvas.active_selector = "RECTANGLE"
        selector = canvas.active_selector

        class _MouseEvent:
            def __init__(self, x, y):
                self.xdata = x
                self.ydata = y

        frame_g, frame_s = plotter.get_merged_features()
        selector.data = np.column_stack((frame_g, frame_s))
        selector.on_select(_MouseEvent(-2.0, -2.0), _MouseEvent(2.0, 2.0))

        # biaplotter hands back one class value per *plotted* point.
        selection_tab.manual_selection_changed(
            np.ones(frame_g.size, dtype=np.uint32)
        )

        layer = plotter.get_selected_layers()[0]
        selection_map = layer.metadata["settings"]["selections"][
            "manual_selections"
        ][selection_tab.selection_id]

        assert selection_map.shape == STACK_SHAPE
        for frame in range(N_FRAMES):
            assert selection_map[frame].any(), f"frame {frame} not labelled"
    finally:
        plotter.close()


# ---------------------------------------------------------------------------
# Animation export
# ---------------------------------------------------------------------------


def test_animation_export_writes_a_gif(make_viewer_model, tmp_path):
    """Rendering and writing a GIF produces one image per requested frame."""
    iio = pytest.importorskip("imageio.v3")

    from napari_phasors._timelapse import export_animation

    viewer = make_viewer_model()
    plotter = make_plotter_with_layer(viewer, create_stack_layer())
    try:
        plotter.frame_context.mode = CURRENT
        options = {
            "include_phasor": True,
            "include_histogram": False,
            "frames": list(range(N_FRAMES)),
            "fps": 5,
        }
        frames = plotter._render_animation_frames(options, histogram=None)
        assert len(frames) == N_FRAMES
        assert frames[0].ndim == 3 and frames[0].shape[2] == 3

        target = tmp_path / "animation.gif"
        assert export_animation(str(target), frames, options["fps"]) is True
        assert target.exists()
        assert len(iio.imread(str(target))) == N_FRAMES
    finally:
        plotter.close()


def test_animation_export_restores_the_starting_frame(make_viewer_model):
    """Exporting must leave the viewer on the frame it started from."""
    viewer = make_viewer_model()
    plotter = make_plotter_with_layer(viewer, create_stack_layer())
    try:
        plotter.frame_context.mode = CURRENT
        plotter.frame_context.index = 2
        plotter._render_animation_frames(
            {
                "include_phasor": True,
                "include_histogram": False,
                "frames": [0, 1, 2, 3],
                "fps": 5,
            },
            histogram=None,
        )
        assert plotter.frame_context.index == 2
    finally:
        plotter.close()


def test_animation_can_include_the_histogram(make_viewer_model):
    """Rendering both figures stacks them side by side in each frame."""
    viewer = make_viewer_model()
    plotter = make_plotter_with_layer(viewer, create_stack_layer())
    try:
        mapping_tab = plotter.phasor_mapping_tab
        plotter.tab_widget.setCurrentWidget(mapping_tab)
        mapping_tab.frequency_input.setText("80")
        mapping_tab.calculate_output_data()
        mapping_tab.plot_lifetime_histogram()

        histogram = plotter._active_histogram_widget()
        assert histogram is mapping_tab.histogram_widget

        plotter.frame_context.mode = CURRENT
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
    finally:
        plotter.close()


def test_animation_export_dialog_options(make_viewer_model, qtbot):
    """The dialog reports a normalised, inclusive frame range."""
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


def test_export_animation_reports_no_frames(tmp_path):
    """Exporting nothing fails cleanly rather than raising."""
    from napari_phasors._timelapse import export_animation

    assert export_animation(str(tmp_path / "empty.gif"), [], 5) is False


def test_combine_frames_pads_to_a_common_height():
    """Side-by-side figures are padded, never scaled."""
    left = np.zeros((10, 4, 3), dtype=np.uint8)
    right = np.zeros((6, 5, 3), dtype=np.uint8)

    assert combine_frames([]) is None
    assert combine_frames([left]).shape == (10, 4, 3)
    assert combine_frames([left, right]).shape == (10, 9, 3)


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


def test_export_animation_click_writes_a_gif(
    make_viewer_model, monkeypatch, tmp_path
):
    """The Export Animation button renders and saves without a dialog."""
    pytest.importorskip("imageio.v3")

    viewer = make_viewer_model()
    plotter = make_plotter_with_layer(viewer, create_stack_layer())
    try:
        plotter.frame_context.mode = CURRENT

        target = tmp_path / "movie"  # no extension: handler must add .gif
        stub = _DialogStub(
            QDialog.Accepted,
            {
                "include_phasor": True,
                "include_histogram": False,
                "frames": list(range(N_FRAMES)),
                "fps": 5,
            },
        )
        monkeypatch.setattr(
            "napari_phasors.plotter.AnimationExportDialog", stub
        )
        _accept_save_dialog(monkeypatch, target)

        plotter._on_export_animation_clicked()

        assert (tmp_path / "movie.gif").exists()
    finally:
        plotter.close()


def test_export_animation_click_needs_per_frame_mode(
    make_viewer_model, monkeypatch
):
    """In pooled mode the handler explains itself instead of exporting."""
    viewer = make_viewer_model()
    plotter = make_plotter_with_layer(viewer, create_stack_layer())
    try:
        messages = []
        monkeypatch.setattr(
            "napari_phasors.plotter.notifications.show_info", messages.append
        )
        # A dialog must never be constructed on this path.
        monkeypatch.setattr(
            "napari_phasors.plotter.AnimationExportDialog",
            lambda *a, **k: pytest.fail("dialog opened in pooled mode"),
        )

        plotter._on_export_animation_clicked()

        assert messages and "Current timepoint" in messages[0]
    finally:
        plotter.close()


def test_export_animation_click_cancelled(make_viewer_model, monkeypatch):
    """Rejecting the options dialog exports nothing."""
    viewer = make_viewer_model()
    plotter = make_plotter_with_layer(viewer, create_stack_layer())
    try:
        plotter.frame_context.mode = CURRENT
        monkeypatch.setattr(
            "napari_phasors.plotter.AnimationExportDialog",
            _DialogStub(QDialog.Rejected),
        )
        monkeypatch.setattr(
            "napari_phasors.plotter.QFileDialog.getSaveFileName",
            staticmethod(
                lambda *a, **k: pytest.fail("save dialog opened after cancel")
            ),
        )

        plotter._on_export_animation_clicked()
    finally:
        plotter.close()


def test_export_animation_click_requires_a_figure(
    make_viewer_model, monkeypatch
):
    """Deselecting both figures warns rather than writing an empty GIF."""
    viewer = make_viewer_model()
    plotter = make_plotter_with_layer(viewer, create_stack_layer())
    try:
        plotter.frame_context.mode = CURRENT
        warnings_seen = []
        monkeypatch.setattr(
            "napari_phasors.plotter.notifications.show_warning",
            warnings_seen.append,
        )
        monkeypatch.setattr(
            "napari_phasors.plotter.AnimationExportDialog",
            _DialogStub(
                QDialog.Accepted,
                {
                    "include_phasor": False,
                    "include_histogram": False,
                    "frames": [0],
                    "fps": 5,
                },
            ),
        )
        monkeypatch.setattr(
            "napari_phasors.plotter.QFileDialog.getSaveFileName",
            staticmethod(lambda *a, **k: pytest.fail("save dialog opened")),
        )

        plotter._on_export_animation_clicked()

        assert warnings_seen
    finally:
        plotter.close()


def test_export_animation_click_no_path_chosen(make_viewer_model, monkeypatch):
    """Dismissing the file dialog exports nothing."""
    viewer = make_viewer_model()
    plotter = make_plotter_with_layer(viewer, create_stack_layer())
    try:
        plotter.frame_context.mode = CURRENT
        monkeypatch.setattr(
            "napari_phasors.plotter.AnimationExportDialog",
            _DialogStub(
                QDialog.Accepted,
                {
                    "include_phasor": True,
                    "include_histogram": False,
                    "frames": [0],
                    "fps": 5,
                },
            ),
        )
        monkeypatch.setattr(
            "napari_phasors.plotter.QFileDialog.getSaveFileName",
            staticmethod(lambda *a, **k: ("", "")),
        )
        monkeypatch.setattr(
            "napari_phasors.plotter.export_animation",
            lambda *a, **k: pytest.fail("exported without a path"),
        )

        plotter._on_export_animation_clicked()
    finally:
        plotter.close()


def test_render_animation_frames_skips_out_of_range(make_viewer_model):
    """Frame indices outside the stack are ignored, not clamped."""
    viewer = make_viewer_model()
    plotter = make_plotter_with_layer(viewer, create_stack_layer())
    try:
        plotter.frame_context.mode = CURRENT
        frames = plotter._render_animation_frames(
            {
                "include_phasor": True,
                "include_histogram": False,
                "frames": [-1, 0, N_FRAMES, N_FRAMES + 5],
                "fps": 5,
            },
            histogram=None,
        )
        assert len(frames) == 1
    finally:
        plotter.close()


def test_render_animation_frames_redraws_the_starting_frame(
    make_viewer_model,
):
    """The frame already displayed still gets rendered."""
    viewer = make_viewer_model()
    plotter = make_plotter_with_layer(viewer, create_stack_layer())
    try:
        plotter.frame_context.mode = CURRENT
        plotter.frame_context.index = 2
        frames = plotter._render_animation_frames(
            {
                "include_phasor": True,
                "include_histogram": False,
                "frames": [2],
                "fps": 5,
            },
            histogram=None,
        )
        assert len(frames) == 1
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


def test_phasor_center_export_per_timepoint(
    make_viewer_model, monkeypatch, tmp_path
):
    """Choosing 'Per timepoint' writes one row per frame."""
    viewer = make_viewer_model()
    plotter = make_plotter_with_layer(viewer, create_stack_layer())
    try:
        target = tmp_path / "centers"  # no extension: handler must add .csv
        _choose_menu_action(monkeypatch, 1)  # "Per timepoint"
        _accept_save_dialog(monkeypatch, target)

        plotter._export_phasor_center_statistics()

        lines = (tmp_path / "centers.csv").read_text().strip().splitlines()
        assert len(lines) == N_FRAMES + 1
        assert lines[0].startswith("Frame,Name,G (center)")
    finally:
        plotter.close()


def test_phasor_center_export_pooled(make_viewer_model, monkeypatch, tmp_path):
    """Choosing 'All timepoints pooled' writes a single row."""
    viewer = make_viewer_model()
    plotter = make_plotter_with_layer(viewer, create_stack_layer())
    try:
        target = tmp_path / "pooled.csv"
        _choose_menu_action(monkeypatch, 0)  # "All timepoints pooled"
        _accept_save_dialog(monkeypatch, target)

        plotter._export_phasor_center_statistics()

        lines = target.read_text().strip().splitlines()
        assert len(lines) == 2
    finally:
        plotter.close()


def test_phasor_center_export_menu_cancelled(make_viewer_model, monkeypatch):
    """Dismissing the menu exports nothing."""
    viewer = make_viewer_model()
    plotter = make_plotter_with_layer(viewer, create_stack_layer())
    try:
        _choose_menu_action(monkeypatch, None)
        monkeypatch.setattr(
            "napari_phasors.plotter.QFileDialog.getSaveFileName",
            staticmethod(lambda *a, **k: pytest.fail("save dialog opened")),
        )

        plotter._export_phasor_center_statistics()
    finally:
        plotter.close()


def test_phasor_center_export_no_path_chosen(make_viewer_model, monkeypatch):
    """Dismissing the file dialog writes nothing."""
    viewer = make_viewer_model()
    plotter = make_plotter_with_layer(viewer, create_stack_layer())
    try:
        _choose_menu_action(monkeypatch, 0)
        monkeypatch.setattr(
            "napari_phasors.plotter.QFileDialog.getSaveFileName",
            staticmethod(lambda *a, **k: ("", "")),
        )
        monkeypatch.setattr(
            "napari_phasors.plotter.write_rows_to_csv",
            lambda *a, **k: pytest.fail("wrote without a path"),
        )

        plotter._export_phasor_center_statistics()
    finally:
        plotter.close()


def test_phasor_center_export_for_2d_data_skips_the_menu(
    make_viewer_model, monkeypatch, tmp_path
):
    """Without a stack axis there is nothing to choose, so no menu appears."""
    viewer = make_viewer_model()
    plotter = make_plotter_with_layer(viewer, create_flat_layer())
    try:
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


def test_phasor_center_export_without_centers_warns(
    make_viewer_model, monkeypatch
):
    """A selection with no computable centers warns instead of writing."""
    viewer = make_viewer_model()
    plotter = make_plotter_with_layer(viewer, create_stack_layer())
    try:
        warnings_seen = []
        monkeypatch.setattr(
            "napari_phasors.plotter.notifications.show_warning",
            warnings_seen.append,
        )
        monkeypatch.setattr(
            plotter, "_phasor_center_statistics_rows", lambda per_frame: []
        )
        monkeypatch.setattr(
            "napari_phasors.plotter.QFileDialog.getSaveFileName",
            staticmethod(lambda *a, **k: pytest.fail("save dialog opened")),
        )
        _choose_menu_action(monkeypatch, 0)

        plotter._export_phasor_center_statistics()

        assert warnings_seen
    finally:
        plotter.close()


# ---------------------------------------------------------------------------
# Guard branches
# ---------------------------------------------------------------------------


def test_frame_callbacks_are_inert_while_closing(make_viewer_model):
    """Queued frame callbacks must not touch a widget that is tearing down."""
    viewer = make_viewer_model()
    plotter = make_plotter_with_layer(viewer, create_stack_layer())
    try:
        plotter.frame_context.mode = CURRENT
        plotter._is_closing = True

        replots = []
        plotter._replot_for_frame_state = lambda: replots.append(True)

        plotter._on_frame_changed(1)
        plotter._on_frame_mode_changed(POOLED)
        plotter._on_frame_axis_changed(0)

        assert replots == []
    finally:
        plotter._is_closing = False
        plotter.close()


def test_frame_changed_is_inert_in_pooled_mode(make_viewer_model):
    """A frame change while pooled changes nothing on screen."""
    viewer = make_viewer_model()
    plotter = make_plotter_with_layer(viewer, create_stack_layer())
    try:
        replots = []
        plotter._replot_for_frame_state = lambda: replots.append(True)

        plotter._on_frame_changed(2)

        assert replots == []
    finally:
        plotter.close()


def test_frame_axis_change_replots_only_per_frame(make_viewer_model):
    """Switching axis matters only when a single frame is displayed."""
    viewer = make_viewer_model()
    plotter = make_plotter_with_layer(
        viewer, create_stack_layer(shape=(2, 3, 5, 6))
    )
    try:
        replots = []
        plotter._replot_for_frame_state = lambda: replots.append(True)

        plotter._on_frame_axis_changed(1)
        assert replots == []

        plotter.frame_context.mode = CURRENT
        plotter._on_frame_axis_changed(0)
        assert replots
    finally:
        plotter.close()


def test_refresh_timelapse_controls_without_a_bar(make_viewer_model):
    """The refresh helper tolerates being called before the bar exists."""
    viewer = make_viewer_model()
    plotter = make_plotter_with_layer(viewer, create_stack_layer())
    try:
        bar = plotter.timelapse_bar
        del plotter.timelapse_bar
        plotter._refresh_timelapse_controls()  # must not raise
        plotter.timelapse_bar = bar
    finally:
        plotter.close()


def test_refresh_frame_dependent_tabs_skips_missing_tabs(make_viewer_model):
    """Tabs that are absent or lack the hook are stepped over."""
    viewer = make_viewer_model()
    plotter = make_plotter_with_layer(viewer, create_stack_layer())
    try:
        original = (
            plotter.phasor_mapping_tab,
            plotter.components_tab,
            plotter.fret_tab,
        )
        plotter.phasor_mapping_tab = None
        # An object that simply has no ``refresh_for_frame_change`` hook.
        plotter.components_tab = object()

        plotter._refresh_frame_dependent_tabs()  # must not raise

        (
            plotter.phasor_mapping_tab,
            plotter.components_tab,
            plotter.fret_tab,
        ) = original
    finally:
        plotter.close()


def test_active_histogram_widget_without_data(make_viewer_model):
    """No analysis run means no histogram to offer the animation export."""
    viewer = make_viewer_model()
    plotter = make_plotter_with_layer(viewer, create_stack_layer())
    try:
        plotter.tab_widget.setCurrentWidget(plotter.settings_tab)
        assert plotter._active_histogram_widget() is None
    finally:
        plotter.close()


def test_deferred_tab_update_skipped_for_a_removed_layer(make_viewer_model):
    """A pending tab update must not look up a layer that is already gone.

    The tab-change event can arrive after the layer was removed; the restore
    paths index the viewer by name, so the update is skipped wholesale.
    """
    viewer = make_viewer_model()
    layer = create_stack_layer()
    plotter = make_plotter_with_layer(viewer, layer)
    try:
        mapping_tab = plotter.phasor_mapping_tab
        mapping_tab._needs_update = True
        restores = []
        mapping_tab._restore_on_layer_change = lambda: restores.append(True)

        # The combobox still reports a layer the viewer no longer holds,
        # which is exactly the state a late tab-change event arrives in.
        plotter.get_primary_layer_name = lambda: "Gone Intensity Image"
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
    finally:
        plotter.close()


def test_layer_phasor_arrays_without_phasor_data(make_viewer_model):
    """A layer with no G/S contributes nothing rather than raising."""
    viewer = make_viewer_model()
    plotter = make_plotter_with_layer(viewer, create_stack_layer())
    try:
        layer = plotter.get_selected_layers()[0]
        layer.metadata.pop("G")

        assert plotter._get_layer_phasor_arrays(layer) is None
        assert list(plotter._iter_layer_gs_arrays()) == []
        assert plotter._phasor_center_statistics_rows(per_frame=True) == []

        plotter.frame_context.mode = CURRENT
        assert plotter._frame_histogram_reference() is None
    finally:
        plotter.close()


def test_frame_histogram_reference_without_a_selection(make_viewer_model):
    """No selected layers means no shared range to compute."""
    viewer = make_viewer_model()
    plotter = make_plotter_with_layer(viewer, create_stack_layer())
    try:
        plotter.frame_context.mode = CURRENT
        plotter.image_layers_checkable_combobox.setCheckedItems([])

        assert plotter._frame_histogram_reference() is None
    finally:
        plotter.close()


def test_frame_histogram_reference_with_all_nan_phasors(make_viewer_model):
    """A fully masked stack yields no usable range."""
    viewer = make_viewer_model()
    layer = create_stack_layer()
    layer.metadata["G"][:] = np.nan
    layer.metadata["S"][:] = np.nan
    plotter = make_plotter_with_layer(viewer, layer)
    try:
        plotter.frame_context.mode = CURRENT
        assert plotter._frame_histogram_reference() is None
    finally:
        plotter.close()


def test_frame_histogram_reference_includes_a_2d_layer(make_viewer_model):
    """A plain 2D layer beside a stack contributes to every frame."""
    viewer = make_viewer_model()
    stack = create_stack_layer(name="Stack")
    flat = create_flat_layer(name="Flat")
    viewer.add_layer(stack)
    viewer.add_layer(flat)

    plotter = PlotterWidget(viewer)
    plotter.image_layers_checkable_combobox.setCheckedItems(
        [stack.name, flat.name]
    )
    plotter._process_layer_selection_change()
    try:
        plotter.frame_context.mode = CURRENT
        reference = plotter._frame_histogram_reference()

        assert reference is not None
        # One stack frame (30 px) plus the whole 2D layer (30 px).
        assert plotter.get_merged_features()[0].size == 2 * 5 * 6
    finally:
        plotter.close()


def test_plot_blanks_when_features_are_unavailable(make_viewer_model):
    """``plot`` blanks the canvas when a frame yields no features at all."""
    viewer = make_viewer_model()
    plotter = make_plotter_with_layer(viewer, create_stack_layer())
    try:
        plotter.frame_context.mode = CURRENT
        plotter.plot()
        assert plotter.canvas_widget.artists['HISTOGRAM2D'].visible is True

        plotter.get_features = lambda: None
        plotter.plot()

        assert plotter.canvas_widget.artists['HISTOGRAM2D'].visible is False
        assert plotter._frame_plot_blanked is True
    finally:
        plotter.close()
