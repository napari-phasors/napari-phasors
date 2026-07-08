"""Tests for the Batch Analysis widget."""

import csv
import os

import numpy as np
import pytest
from napari.layers import Image

from napari_phasors._batch_analysis import (
    BatchAnalysisWidget,
    BatchPipeline,
    BatchReadOptionsWidget,
    CopySettingsDialog,
    _apply_component_fraction,
    _apply_fret,
    _apply_image_mask,
    _apply_phasor_mapping,
    _apply_selection,
    _cursor_rgba,
    _load_mask_array,
    _make_output_image,
    _reversed_colormap,
    _select_harmonic_arrays,
    _selection_statistics,
    _store_plot_settings,
    apply_pipeline,
    default_component_label_style,
    match_extension,
    parse_harmonics,
    scan_folder,
    supported_extensions,
)
from napari_phasors._reader import napari_get_reader
from napari_phasors._synthetic_generator import (
    make_intensity_layer_with_phasors,
    make_raw_flim_data,
)
from napari_phasors._utils import (
    apply_filter_and_threshold,
    read_ome_tiff_settings,
)
from napari_phasors._writer import write_ome_tiff


def _make_phasor_layer(name="FLIM data", harmonic=None):
    raw = make_raw_flim_data(time_constants=[0.1, 1, 10])
    return make_intensity_layer_with_phasors(raw, harmonic=harmonic, name=name)


def _select_plot_type(combo, key):
    """Select a phasor-plot type combobox entry by its canonical key.

    The combobox shows friendly labels ("Density Plot (2D Histogram)", ...)
    while storing the canonical key ("Histogram", "Scatter", "Contour",
    "None") as the item's userData.
    """
    combo.setCurrentIndex(combo.findData(key))


# -- Pure helpers ----------------------------------------------------------


def test_supported_extensions_orders_specific_first():
    exts = supported_extensions()
    assert ".ome.tif" in exts
    assert ".fbd" in exts
    # Longer extensions sort before shorter ones.
    assert exts.index(".ome.tif") < exts.index(".tif")


def test_match_extension_prefers_longest_suffix():
    assert match_extension("image.ome.tif") == ".ome.tif"
    assert match_extension("image.tif") == ".tif"
    assert match_extension("data.fbd") == ".fbd"
    assert match_extension("notes.txt") is None


def test_parse_harmonics():
    assert parse_harmonics("1, 2") == [1, 2]
    assert parse_harmonics("3") == [3]
    assert parse_harmonics("all") == "all"
    assert parse_harmonics("") is None


def test_scan_folder_respects_recursion(tmp_path):
    (tmp_path / "a.ome.tif").write_bytes(b"x")
    (tmp_path / "b.fbd").write_bytes(b"x")
    (tmp_path / "ignore.txt").write_bytes(b"x")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.ome.tif").write_bytes(b"x")

    flat = scan_folder(str(tmp_path), recursive=False)
    assert set(flat) == {".ome.tif", ".fbd"}
    assert len(flat[".ome.tif"]) == 1

    deep = scan_folder(str(tmp_path), recursive=True)
    assert len(deep[".ome.tif"]) == 2


def test_mask_phasor_arrays_same_ndim():
    """Test _mask_phasor_arrays when array has same ndim as data."""
    layer = Image(np.zeros((2, 2)), name="test")
    layer.metadata["G"] = np.ones((2, 2))
    layer.metadata["S"] = np.ones((2, 2))
    from napari_phasors._batch_analysis import _apply_image_mask

    mask_invalid = np.array([[True, False], [False, True]])
    # invert=False means mask > 0 is kept, mask <= 0 is invalid.
    # mask_invalid being boolean: True is kept, False is invalid.
    _apply_image_mask(layer, mask_invalid)
    assert np.isnan(layer.metadata["G"][0, 1])
    assert layer.metadata["G"][0, 0] == 1.0


def test_select_harmonic_arrays_missing_gs():
    """Test _select_harmonic_arrays when G or S is missing."""
    layer = Image(np.zeros((2, 2)), name="test")
    layer.metadata["G"] = np.ones((2, 2))
    layer.metadata["S"] = None
    real, imag = _select_harmonic_arrays(layer, 1)
    assert real is None
    assert imag is None


# -- Widget behaviour ------------------------------------------------------


def test_format_combobox_populated(qtbot, make_viewer_model, tmp_path):
    (tmp_path / "a.ome.tif").write_bytes(b"x")
    (tmp_path / "b.ome.tif").write_bytes(b"x")
    (tmp_path / "c.fbd").write_bytes(b"x")

    widget = BatchAnalysisWidget(make_viewer_model())
    qtbot.addWidget(widget)

    widget._input_folder = str(tmp_path)
    widget._rescan()

    labels = [
        widget.format_combobox.itemText(i)
        for i in range(widget.format_combobox.count())
    ]
    assert any("(2 files)" in label for label in labels)
    assert widget.format_combobox.count() == 2


def test_layer_events_refresh_reference_combo_via_bound_method(
    qtbot, make_viewer_model
):
    """``viewer.layers.events`` are connected to a bound method (not a
    lambda) so ``closeEvent`` can disconnect them; this exercises that the
    connection actually refreshes the UI on insert/remove after
    construction (not just the initial population call in ``__init__``).
    """
    viewer = make_viewer_model()
    widget = BatchAnalysisWidget(viewer)
    qtbot.addWidget(widget)

    assert widget.calib_reference_combo.count() == 0

    layer = _make_phasor_layer(name="new_layer")
    viewer.add_layer(layer)

    labels = [
        widget.calib_reference_combo.itemText(i)
        for i in range(widget.calib_reference_combo.count())
    ]
    assert layer.name in labels

    viewer.layers.remove(layer)
    assert widget.calib_reference_combo.count() == 0


def test_get_reader_options_with_dynamic_kwargs(qtbot, make_viewer_model):
    widget = BatchAnalysisWidget(make_viewer_model())
    qtbot.addWidget(widget)

    widget.read_options_widget.set_extension(".fbd")
    options = widget.read_options_widget.get_reader_options()
    assert options == {"frame": -1, "keepdims": False, "channel": None}

    widget.read_options_widget._add_kwarg_row("laser_factor", "2.5")
    options = widget.read_options_widget.get_reader_options()
    assert options["laser_factor"] == 2.5


# -- Pipeline --------------------------------------------------------------


def test_apply_pipeline_filter_matches_direct_call():
    layer = _make_phasor_layer()
    reference = _make_phasor_layer()

    filter_kwargs = {
        "filter_method": "median",
        "size": 3,
        "repeat": 1,
        "threshold_method": "manual",
        "threshold": 1.0,
    }
    pipeline = BatchPipeline(filter=filter_kwargs)
    apply_pipeline(layer, pipeline)

    apply_filter_and_threshold(reference, **filter_kwargs)

    np.testing.assert_array_equal(layer.data, reference.data)
    np.testing.assert_array_equal(layer.metadata["G"], reference.metadata["G"])


def test_apply_settings_to_ui_from_layer(qtbot, make_viewer_model):
    widget = BatchAnalysisWidget(make_viewer_model())
    qtbot.addWidget(widget)

    settings = {
        "filter": {"method": "median", "size": 5, "repeat": 2},
        "threshold": 1.0,
        "threshold_method": "manual",
    }
    widget._apply_settings_to_ui(settings)

    assert widget.filter_group.isChecked()
    assert widget.filter_method_combo.currentText() == "Median"
    assert widget.median_size_spin.value() == 5
    assert widget.median_repeat_spin.value() == 2
    assert widget.threshold_method_combo.currentText() == "Manual"
    assert widget.threshold_min_spin.value() == 1.0

    pipeline = widget.build_pipeline([1, 2])
    assert pipeline.filter is not None
    assert pipeline.filter["filter_method"] == "median"
    assert pipeline.filter["size"] == 5
    assert pipeline.filter["repeat"] == 2

    # The resolved pipeline matches a direct filter call.
    target = _make_phasor_layer()
    reference = _make_phasor_layer()
    apply_pipeline(target, pipeline)
    apply_filter_and_threshold(
        reference,
        filter_method="median",
        size=5,
        repeat=2,
        threshold=1.0,
        threshold_method="manual",
    )
    np.testing.assert_array_equal(
        target.metadata["G"], reference.metadata["G"]
    )


def test_calibration_same_for_all(qtbot, make_viewer_model):
    viewer = make_viewer_model()
    reference = _make_phasor_layer(name="calibration")
    viewer.add_layer(reference)

    widget = BatchAnalysisWidget(viewer)
    qtbot.addWidget(widget)
    widget.calibration_group.setChecked(True)
    idx = widget.calib_source_combo.findData("same")
    widget.calib_source_combo.setCurrentIndex(idx)
    widget.calib_reference_combo.setCurrentText(reference.name)
    widget.calib_frequency_spin.setText("80.0")
    widget.calib_lifetime_spin.setValue(2.0)

    calibration_map = widget._resolve_calibration_map([1, 2])
    assert "*" in calibration_map
    assert "phi_zero" in calibration_map["*"]

    target = _make_phasor_layer()
    before = target.metadata["G"].copy()
    pipeline = widget.build_pipeline([1, 2])
    pipeline.calibration = widget._calibration_for(
        "any.ome.tif", calibration_map
    )
    apply_pipeline(target, pipeline)
    assert target.metadata["settings"]["calibrated"] is True
    assert not np.array_equal(before, target.metadata["G"])


def test_calibration_per_subfolder(qtbot, make_viewer_model, tmp_path):
    in_root = tmp_path / "in"
    (in_root / "cond1").mkdir(parents=True)
    (in_root / "cond2").mkdir(parents=True)
    write_ome_tiff(
        str(in_root / "cond1" / "a.ome.tif"), _make_phasor_layer(name="a")
    )
    write_ome_tiff(
        str(in_root / "cond2" / "b.ome.tif"), _make_phasor_layer(name="b")
    )
    # References live outside the scanned tree (not treated as input).
    refs = tmp_path / "refs"
    refs.mkdir()
    ref1 = refs / "ref1.ome.tif"
    ref2 = refs / "ref2.ome.tif"
    write_ome_tiff(str(ref1), _make_phasor_layer(name="r1"))
    write_ome_tiff(str(ref2), _make_phasor_layer(name="r2"))

    widget = BatchAnalysisWidget(make_viewer_model())
    qtbot.addWidget(widget)
    widget._input_folder = str(in_root)
    widget.subfolders_checkbox.setChecked(True)
    widget._rescan()
    idx = widget.format_combobox.findData(".ome.tif")
    widget.format_combobox.setCurrentIndex(idx)

    widget.calibration_group.setChecked(True)
    src = widget.calib_source_combo.findData("subfolder")
    widget.calib_source_combo.setCurrentIndex(src)
    widget.calib_frequency_spin.setText("80.0")
    widget.calib_lifetime_spin.setValue(2.0)

    assert set(widget._subfolder_ref_edits) == {"cond1", "cond2"}
    widget._subfolder_ref_edits["cond1"].setText(str(ref1))
    widget._subfolder_ref_edits["cond2"].setText(str(ref2))

    calibration_map = widget._resolve_calibration_map([1, 2])
    assert set(calibration_map) == {"cond1", "cond2"}
    assert "phi_zero" in calibration_map["cond1"]

    # Files are matched to their own subfolder's calibration.
    p1 = widget._calibration_for(
        str(in_root / "cond1" / "a.ome.tif"), calibration_map
    )
    p2 = widget._calibration_for(
        str(in_root / "cond2" / "b.ome.tif"), calibration_map
    )
    assert p1 is calibration_map["cond1"]
    assert p2 is calibration_map["cond2"]


def test_collect_components_linear_then_fit(qtbot, make_viewer_model):
    widget = BatchAnalysisWidget(make_viewer_model())
    qtbot.addWidget(widget)
    widget.components_group.setChecked(True)

    # Two components default -> Linear Projection available.
    widget.analysis_type_combo.setCurrentText("Linear Projection")
    components = widget.build_pipeline([1, 2]).components
    assert components["analysis_type"] == "linear"
    assert len(components["names"]) == 2

    # Adding a third component forces the Component Fit path.
    widget._add_component_row("Third", 0.5, 0.5)
    components = widget.build_pipeline([1, 2]).components
    assert components["analysis_type"] == "fit"
    assert len(components["component_real"]) == 3


def test_n_component_fit_pipeline():
    layer = _make_phasor_layer(harmonic=[1, 2])
    pipeline = BatchPipeline(
        components={
            "analysis_type": "fit",
            "names": ["A", "B", "C"],
            "component_real": [0.1, 0.5, 0.8],
            "component_imag": [0.2, 0.4, 0.3],
            "harmonic": 1,
        }
    )
    extra_layers = apply_pipeline(layer, pipeline)
    assert len(extra_layers) == 3
    assert all("fraction" in lyr.name for lyr in extra_layers)


def test_required_component_harmonics():
    from napari_phasors._batch_analysis import required_component_harmonics

    assert required_component_harmonics(2) == 1
    assert required_component_harmonics(3) == 1
    assert required_component_harmonics(4) == 2
    assert required_component_harmonics(5) == 2
    assert required_component_harmonics(6) == 3
    assert required_component_harmonics(7) == 3


def test_multiharmonic_component_fit_pipeline():
    """A >3-component fit stacks per-harmonic locations and component arrays."""
    layer = _make_phasor_layer(harmonic=[1, 2])
    pipeline = BatchPipeline(
        components={
            "analysis_type": "fit",
            "names": ["A", "B", "C", "D"],
            "harmonics": [1, 2],
            "component_real": [
                [0.1, 0.4, 0.7, 0.9],
                [0.05, 0.2, 0.4, 0.6],
            ],
            "component_imag": [
                [0.2, 0.4, 0.3, 0.1],
                [0.1, 0.25, 0.3, 0.2],
            ],
            "harmonic": 1,
        }
    )
    extra_layers = apply_pipeline(layer, pipeline)
    assert len(extra_layers) == 4
    assert all("fraction" in lyr.name for lyr in extra_layers)


def test_collect_components_multiharmonic(qtbot, make_viewer_model):
    widget = BatchAnalysisWidget(make_viewer_model())
    qtbot.addWidget(widget)
    widget.components_group.setChecked(True)

    # Four components -> requires two harmonics of locations.
    widget._add_component_row("C", 0.7, 0.3)
    widget._add_component_row("D", 0.9, 0.1)
    for i, (g, s) in enumerate(
        [(0.1, 0.2), (0.4, 0.4), (0.7, 0.3), (0.9, 0.1)]
    ):
        widget._component_rows[i]["g"].setText(str(g))
        widget._component_rows[i]["s"].setText(str(s))
        widget._store_component_coord(widget._component_rows[i])

    # Only harmonic 1 provided -> collecting raises a clear error.
    with pytest.raises(ValueError, match="requires component locations"):
        widget.build_pipeline([1, 2])

    # Provide harmonic-2 locations via the harmonic selector.
    widget.components_harmonic_spin.setValue(2)
    assert widget._component_current_harmonic == 2
    for i, (g, s) in enumerate(
        [(0.05, 0.1), (0.2, 0.25), (0.4, 0.3), (0.6, 0.2)]
    ):
        widget._component_rows[i]["g"].setText(str(g))
        widget._component_rows[i]["s"].setText(str(s))
        widget._store_component_coord(widget._component_rows[i])

    components = widget.build_pipeline([1, 2]).components
    assert components["analysis_type"] == "fit"
    assert components["harmonics"] == [1, 2]
    assert np.array(components["component_real"]).shape == (2, 4)
    assert np.array(components["component_imag"]).shape == (2, 4)

    # Switching back to harmonic 1 restores its stored locations in the fields.
    widget.components_harmonic_spin.setValue(1)
    assert widget._component_rows[0]["g"].text() == "0.1"


def test_phasor_mapping_pipeline():
    for output_type in (
        "Apparent Phase Lifetime",
        "Normal Lifetime",
        "Phase",
        "Modulation",
    ):
        layer = _make_phasor_layer(harmonic=[1, 2])
        pipeline = BatchPipeline(
            mapping={
                "output_type": output_type,
                "frequency": 80.0,
                "harmonic": 1,
            }
        )
        extra_layers = apply_pipeline(layer, pipeline)
        assert len(extra_layers) == 1
        assert extra_layers[0].name.startswith(output_type)
        assert extra_layers[0].data.shape == layer.data.shape


def test_fret_pipeline():
    layer = _make_phasor_layer(harmonic=[1, 2])
    pipeline = BatchPipeline(
        fret={
            "donor_lifetime": 2.0,
            "frequency": 80.0,
            "harmonic": 1,
            "donor_background": 0.1,
            "donor_fretting": 1.0,
            "background_real": 0.0,
            "background_imag": 0.0,
        }
    )
    extra_layers = apply_pipeline(layer, pipeline)
    assert len(extra_layers) == 1
    assert extra_layers[0].name.startswith("FRET efficiency")
    finite = extra_layers[0].data[np.isfinite(extra_layers[0].data)]
    assert np.all((finite >= 0) & (finite <= 1))


def test_selection_pipeline():
    layer = _make_phasor_layer(harmonic=[1, 2])
    pipeline = BatchPipeline(
        selection={
            "harmonic": 1,
            "cursors": [
                {"g": 0.5, "s": 0.4, "radius": 0.5},
                {"g": 0.1, "s": 0.2, "radius": 0.5},
            ],
        }
    )
    extra_layers = apply_pipeline(layer, pipeline)
    assert len(extra_layers) == 1
    selection = extra_layers[0]
    assert selection.name.startswith("Cursor selection")
    assert selection.data.shape == layer.data.shape
    # Labels are 0 (unselected) plus one id per cursor that matched.
    assert set(np.unique(selection.data)).issubset({0, 1, 2})


def test_selection_statistics_manual():
    """Manual cursor stats give per-cursor pixel counts and percentages."""
    layer = _make_phasor_layer(harmonic=[1, 2])
    selection = {
        "harmonic": 1,
        "mode": "manual",
        "cursors": [
            # A huge cursor capturing every finite pixel, plus a tiny one.
            {"type": "circular", "g": 0.5, "s": 0.25, "radius": 10.0},
            {"type": "circular", "g": 0.0, "s": 0.0, "radius": 0.001},
        ],
    }
    selection_map = _apply_selection(layer, selection)[0].data
    rows = _selection_statistics(layer, selection, selection_map)
    assert len(rows) == 2
    assert rows[0]["region"] == "Cursor 1"
    assert rows[0]["percent"] > 99.0  # captures essentially all valid pixels
    assert rows[1]["count"] == 0


def test_selection_statistics_cluster():
    """Cluster stats sum to 100% of the valid pixels across clusters."""
    layer = _make_phasor_layer(harmonic=[1, 2])
    selection = {
        "harmonic": 1,
        "mode": "cluster",
        "cluster": {"clusters": 2, "sigma": 2.0},
    }
    selection_map = _apply_selection(layer, selection)[0].data
    rows = _selection_statistics(layer, selection, selection_map)
    assert rows
    assert all(row["type"] == "cluster" for row in rows)
    assert abs(sum(row["percent"] for row in rows) - 100.0) < 1e-6


def test_selection_stats_csv_export(qtbot, make_viewer_model, tmp_path):
    """Enabling selection stats writes one CSV with per-cursor rows."""
    in_root = tmp_path / "in"
    in_root.mkdir()
    out_root = tmp_path / "out"
    out_root.mkdir()
    for name in ("a", "b"):
        write_ome_tiff(
            str(in_root / f"{name}.ome.tif"),
            _make_phasor_layer(name=name, harmonic=[1, 2]),
        )

    widget = BatchAnalysisWidget(make_viewer_model())
    qtbot.addWidget(widget)
    widget._input_folder = str(in_root)
    widget._rescan()
    idx = widget.format_combobox.findData(".ome.tif")
    widget.format_combobox.setCurrentIndex(idx)
    widget.harmonics_edit.setText("1, 2")
    widget._export_folder = str(out_root)
    widget.export_ometiff_checkbox.setChecked(False)
    widget.selection_group.setChecked(True)
    widget._add_cursor_row(0.3, 0.3, 0.1)  # two cursors total
    widget.selection_stats_checkbox.setChecked(True)
    # Selection stats alone are a valid output to run.
    assert widget._has_extra_outputs()

    widget.run_batch()

    csv_path = (
        out_root
        / "Combined analysis"
        / "Statistics"
        / "selection_statistics.csv"
    )
    assert csv_path.exists()
    with open(csv_path, newline="") as handle:
        rows = list(csv.reader(handle))
    # Header + (2 files x 2 cursors) rows.
    assert len(rows) == 1 + 4
    assert rows[0][:3] == ["File", "Region", "Type"]


def test_phasor_export_toggle_overlay_label(qtbot, make_viewer_model):
    widget = BatchAnalysisWidget(make_viewer_model())
    qtbot.addWidget(widget)
    assert (
        widget.components_plot_toggle.text()
        == "Export Phasor Plot with Components Overlay (PNG)"
    )
    assert "Selection Overlay" in widget.selection_plot_toggle.text()
    # The base phasor plot is governed by the individual / combined export
    # checkboxes (both on by default), not a separate master toggle.
    assert widget.plot_individual_checkbox.isChecked()
    assert widget.plot_combined_checkbox.isChecked()


def test_component_labels_hidden_by_default():
    assert default_component_label_style()["show_labels"] is False


def test_collect_selection(qtbot, make_viewer_model):
    widget = BatchAnalysisWidget(make_viewer_model())
    qtbot.addWidget(widget)
    widget.selection_group.setChecked(True)
    widget.selection_harmonic_spin.setValue(2)
    widget._add_cursor_row(0.3, 0.3, 0.1)

    selection = widget.build_pipeline([1, 2]).selection
    assert selection["harmonic"] == 2
    assert selection["mode"] == "manual"
    assert len(selection["cursors"]) == 2
    cursor = selection["cursors"][1]
    assert cursor["type"] == "circular"
    assert (cursor["g"], cursor["s"], cursor["radius"]) == (0.3, 0.3, 0.1)
    assert "color" in cursor


def test_selection_elliptic_and_polar_pipeline():
    layer = _make_phasor_layer(harmonic=[1, 2])
    pipeline = BatchPipeline(
        selection={
            "harmonic": 1,
            "cursors": [
                {
                    "type": "elliptic",
                    "g": 0.4,
                    "s": 0.3,
                    "radius": 0.6,
                    "radius_minor": 0.3,
                    "angle": 0.0,
                },
                {
                    "type": "polar",
                    "phase_min": 0.0,
                    "phase_max": 2.0,
                    "modulation_min": 0.0,
                    "modulation_max": 1.0,
                },
            ],
        }
    )
    extra_layers = apply_pipeline(layer, pipeline)
    assert len(extra_layers) == 1
    assert extra_layers[0].data.shape == layer.data.shape
    assert set(np.unique(extra_layers[0].data)).issubset({0, 1, 2})


def test_collect_selection_cursor_types(qtbot, make_viewer_model):
    widget = BatchAnalysisWidget(make_viewer_model())
    qtbot.addWidget(widget)
    widget.selection_group.setChecked(True)
    widget._cursor_rows[0]["type"].setCurrentText("Elliptic")
    widget._add_cursor_row()
    widget._cursor_rows[1]["type"].setCurrentText("Polar")

    cursors = widget.build_pipeline([1, 2]).selection["cursors"]
    assert cursors[0]["type"] == "elliptic"
    assert "radius_minor" in cursors[0] and "angle" in cursors[0]
    assert cursors[1]["type"] == "polar"
    assert "phase_min" in cursors[1] and "modulation_max" in cursors[1]


def test_component_lifetime_sets_gs(qtbot, make_viewer_model):
    widget = BatchAnalysisWidget(make_viewer_model())
    qtbot.addWidget(widget)
    widget.components_group.setChecked(True)
    widget.components_frequency_spin.setText("80.0")
    widget.components_harmonic_spin.setValue(1)
    row = widget._component_rows[0]
    row["lifetime"].setText("4.0")
    widget._on_component_lifetime_edited(row)
    # phasor_from_lifetime(80, 4) lands inside the universal semicircle.
    assert 0.0 < float(row["g"].text()) < 1.0
    assert 0.0 < float(row["s"].text()) < 0.6


def test_filter_param_visibility(qtbot, make_viewer_model):
    widget = BatchAnalysisWidget(make_viewer_model())
    qtbot.addWidget(widget)
    widget.filter_method_combo.setCurrentText("Median")
    assert not widget.median_filter_widget.isHidden()
    assert widget.wavelet_filter_widget.isHidden()
    widget.filter_method_combo.setCurrentText("Wavelet")
    assert widget.median_filter_widget.isHidden()
    assert not widget.wavelet_filter_widget.isHidden()
    widget.threshold_method_combo.setCurrentText("Manual")
    assert not widget.threshold_manual_widget.isHidden()
    widget.threshold_method_combo.setCurrentText("Otsu")
    assert widget.threshold_manual_widget.isHidden()


def test_collect_mapping_and_fret(qtbot, make_viewer_model):
    widget = BatchAnalysisWidget(make_viewer_model())
    qtbot.addWidget(widget)

    widget.mapping_group.setChecked(True)
    widget.mapping_output_combo.setCheckedItems(["Normal Lifetime", "Phase"])
    widget.mapping_frequency_spin.setText("40.0")
    widget.mapping_harmonic_spin.setValue(2)

    widget.fret_group.setChecked(True)
    widget.fret_donor_lifetime_spin.setValue(3.5)

    pipeline = widget.build_pipeline([1, 2])
    assert pipeline.mapping["output_types"] == ["Normal Lifetime", "Phase"]
    assert pipeline.mapping["frequency"] == 40.0
    assert pipeline.mapping["harmonic"] == 2
    assert "colormap" in pipeline.mapping
    assert pipeline.fret["donor_lifetime"] == 3.5
    assert pipeline.fret["harmonic"] == 1


# -- Round 6: mapping multi-output / mesh, streaming, threading -------------


def test_mapping_multi_output_pipeline():
    layer = _make_phasor_layer(harmonic=[1, 2])
    pipeline = BatchPipeline(
        mapping={
            "output_types": [
                "Apparent Phase Lifetime",
                "Phase",
                "Modulation",
            ],
            "frequency": 80.0,
            "harmonic": 1,
        }
    )
    out = apply_pipeline(layer, pipeline)
    names = [lyr.name for lyr in out]
    assert len(out) == 3
    assert any(n.startswith("Phase:") for n in names)
    assert any(n.startswith("Modulation:") for n in names)


def test_mapping_mesh_plot_jobs(qtbot, make_viewer_model):
    widget = BatchAnalysisWidget(make_viewer_model())
    qtbot.addWidget(widget)
    widget.mapping_group.setChecked(True)
    widget.mapping_plot_toggle.setChecked(True)
    widget.mapping_color_by_combo.setCurrentText("Phase")
    widget.mapping_mesh_phase_checkbox.setChecked(True)
    widget.mapping_mesh_mod_checkbox.setChecked(True)

    jobs = widget._collect_plot_jobs(widget.build_pipeline([1, 2]))
    suffixes = [job["suffix"] for job in jobs]
    assert "mapping_phasor" in suffixes
    assert "mapping_phasor_phase_mesh" in suffixes
    assert "mapping_phasor_modulation_mesh" in suffixes


def test_mapping_mesh_ranges_collected(qtbot, make_viewer_model):
    """Customizable phase/modulation ranges flow into the mesh plot jobs."""
    widget = BatchAnalysisWidget(make_viewer_model())
    qtbot.addWidget(widget)
    widget.mapping_group.setChecked(True)
    widget.mapping_plot_toggle.setChecked(True)
    # Manual ranges (Auto off) flow through verbatim.
    widget.mapping_range_auto_checkbox.setChecked(False)
    widget.mapping_mesh_phase_checkbox.setChecked(True)
    widget.mapping_phase_min_spin.setValue(0.2)
    widget.mapping_phase_max_spin.setValue(1.1)
    widget.mapping_mod_min_spin.setValue(0.3)
    widget.mapping_mod_max_spin.setValue(0.9)
    widget.mapping_mesh_clip_checkbox.setChecked(False)

    mapping = widget._collect_mapping()
    assert mapping["mesh_phase_range"] == (0.2, 1.1)
    assert mapping["mesh_modulation_range"] == (0.3, 0.9)
    assert mapping["mesh_clip_semicircle"] is False

    jobs = widget._collect_plot_jobs(widget.build_pipeline([1, 2]))
    mesh_jobs = [j for j in jobs if j["overlay"] and j["overlay"].get("mesh")]
    assert mesh_jobs
    for job in mesh_jobs:
        assert job["overlay"]["mesh_phase_range"] == (0.2, 1.1)
        assert job["overlay"]["mesh_modulation_range"] == (0.3, 0.9)
        assert job["overlay"]["mesh_clip_semicircle"] is False


def test_draw_phasor_mesh_keeps_square_aspect():
    """The shared mesh helper restores a 1:1 data aspect (no distortion)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from napari_phasors.phasor_mapping_tab import draw_phasor_mesh

    fig, ax = plt.subplots()
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 0.7)
    image = draw_phasor_mesh(
        ax,
        "Phase",
        semicircle=True,
        phase_range=(0.0, 1.0),
        modulation_range=(0.2, 1.0),
        clip_semicircle=True,
    )
    assert image is not None
    assert ax.get_aspect() == 1.0
    plt.close(fig)


def test_progress_is_noop_off_main_thread(qtbot):
    import threading

    from napari_phasors._utils import _NullProgress, show_activity_progress

    result = {}

    def worker():
        result["pbr"] = show_activity_progress("x", total=1)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    # Off the GUI thread a no-op progress is returned (no Qt objects created),
    # so batch worker threads don't trigger Qt timer/parent warnings.
    assert isinstance(result["pbr"], _NullProgress)


def test_threaded_run_processes_all_files(qtbot, make_viewer_model, tmp_path):
    in_root = tmp_path / "in"
    in_root.mkdir()
    out_root = tmp_path / "out"
    out_root.mkdir()
    for name in ("a", "b", "c"):
        write_ome_tiff(
            str(in_root / f"{name}.ome.tif"),
            _make_phasor_layer(name=name, harmonic=[1, 2]),
        )

    widget = BatchAnalysisWidget(make_viewer_model())
    qtbot.addWidget(widget)
    widget._input_folder = str(in_root)
    widget._rescan()
    idx = widget.format_combobox.findData(".ome.tif")
    widget.format_combobox.setCurrentIndex(idx)
    widget._export_folder = str(out_root)
    widget.export_ometiff_checkbox.setChecked(True)
    widget.threads_spin.setValue(3)
    widget.suffix_edit.setText("_analyzed")

    widget.run_batch()

    assert len(list(out_root.rglob("*_analyzed.ome.tif"))) == 3
    assert "3/3 files processed" in widget.status_label.text()


def test_streaming_aggregation_outputs(qtbot, make_viewer_model, tmp_path):
    in_root = tmp_path / "in"
    in_root.mkdir()
    out_root = tmp_path / "out"
    out_root.mkdir()
    write_ome_tiff(
        str(in_root / "a.ome.tif"),
        _make_phasor_layer(name="a", harmonic=[1, 2]),
    )
    write_ome_tiff(
        str(in_root / "b.ome.tif"),
        _make_phasor_layer(name="b", harmonic=[1, 2]),
    )

    widget = BatchAnalysisWidget(make_viewer_model())
    qtbot.addWidget(widget)
    widget._input_folder = str(in_root)
    widget._rescan()
    idx = widget.format_combobox.findData(".ome.tif")
    widget.format_combobox.setCurrentIndex(idx)
    widget.harmonics_edit.setText("1, 2")
    widget._export_folder = str(out_root)
    widget.export_ometiff_checkbox.setChecked(False)

    widget.streaming_checkbox.setChecked(True)
    widget._group_config = {
        "mode": "Merged",
        "assignments": {},
        "group_names": {},
        "group_colors": {},
        "show_sd": False,
        "central_tendency": "None",
        "show_legend": True,
    }
    _select_plot_type(widget._plot_combined_controls["type"], "Contour")
    widget.mapping_group.setChecked(True)
    widget.mapping_output_combo.setCheckedItems(["Normal Lifetime"])
    widget.mapping_export_controls["stats"].setChecked(True)

    widget.run_batch()

    # Streaming combined contour (disk-backed) per harmonic.
    assert (
        out_root
        / "Combined analysis"
        / "Phasor Plots"
        / "combined_contour_H1.png"
    ).exists()
    # Grouped statistics computed exactly from the spilled data.
    grouped_csvs = list(
        out_root.rglob("phasor_mapping_grouped_statistics.csv")
    )
    assert grouped_csvs
    with open(grouped_csvs[0], newline="") as handle:
        rows = list(csv.reader(handle))
    # Header + at least one group row with a positive pixel count.
    assert len(rows) >= 2
    assert int(rows[1][-1]) > 0


def _run_grouped_stats(widget_factory, tmp_path, streaming):
    """Run a small grouped mapping batch and return the stats CSV rows."""
    in_root = tmp_path / ("in_s" if streaming else "in_x")
    in_root.mkdir()
    out_root = tmp_path / ("out_s" if streaming else "out_x")
    out_root.mkdir()
    for name in ("a", "b", "c"):
        write_ome_tiff(
            str(in_root / f"{name}.ome.tif"),
            _make_phasor_layer(name=name, harmonic=[1, 2]),
        )
    widget = widget_factory()
    widget._input_folder = str(in_root)
    widget._rescan()
    idx = widget.format_combobox.findData(".ome.tif")
    widget.format_combobox.setCurrentIndex(idx)
    widget.harmonics_edit.setText("1, 2")
    widget._export_folder = str(out_root)
    widget.export_ometiff_checkbox.setChecked(False)
    widget.streaming_checkbox.setChecked(streaming)
    widget._group_config = {
        "mode": "Grouped",
        "assignments": {"a.ome.tif": 1, "b.ome.tif": 1, "c.ome.tif": 2},
        "group_names": {1: "A", 2: "B"},
        "group_colors": {1: "#ff0000", 2: "#0000ff"},
        "show_sd": False,
        "central_tendency": "None",
        "show_legend": True,
    }
    widget.mapping_group.setChecked(True)
    widget.mapping_output_combo.setCheckedItems(["Normal Lifetime"])
    widget.mapping_export_controls["stats"].setChecked(True)
    widget.run_batch()
    csv_path = next(
        iter(out_root.rglob("phasor_mapping_grouped_statistics.csv"))
    )
    with open(csv_path, newline="") as handle:
        rows = list(csv.reader(handle))
    return {(r[0], r[1]): tuple(r[2:]) for r in rows[1:]}


def test_streaming_matches_exact_stats(qtbot, make_viewer_model, tmp_path):
    def factory():
        widget = BatchAnalysisWidget(make_viewer_model())
        qtbot.addWidget(widget)
        return widget

    exact = _run_grouped_stats(factory, tmp_path, streaming=False)
    streamed = _run_grouped_stats(factory, tmp_path, streaming=True)
    # Streaming spills to disk but computes stats on the full exact data, so
    # mean/median/std/center-of-mass/pixel-count match exactly.
    assert exact == streamed
    assert exact  # non-empty


def test_grouped_histogram_export_with_sd(qtbot, make_viewer_model, tmp_path):
    """Grouped histogram export writes a PNG per analysis output with SD on."""
    in_root = tmp_path / "in"
    in_root.mkdir()
    out_root = tmp_path / "out"
    out_root.mkdir()
    for name in ("a", "b", "c"):
        write_ome_tiff(
            str(in_root / f"{name}.ome.tif"),
            _make_phasor_layer(name=name, harmonic=[1, 2]),
        )

    widget = BatchAnalysisWidget(make_viewer_model())
    qtbot.addWidget(widget)
    widget._input_folder = str(in_root)
    widget._rescan()
    idx = widget.format_combobox.findData(".ome.tif")
    widget.format_combobox.setCurrentIndex(idx)
    widget.harmonics_edit.setText("1, 2")
    widget._export_folder = str(out_root)
    widget.export_ometiff_checkbox.setChecked(False)
    # Two files in group 1, one in group 2, SD shading on.
    widget._group_config = {
        "mode": "Grouped",
        "assignments": {"a.ome.tif": 1, "b.ome.tif": 1, "c.ome.tif": 2},
        "group_names": {1: "A", 2: "B"},
        "group_colors": {1: "#ff0000", 2: "#0000ff"},
        "show_sd": True,
        "central_tendency": "None",
        "show_legend": True,
        "white_background": False,
        "smooth_curves": True,
    }
    widget.mapping_group.setChecked(True)
    widget.mapping_output_combo.setCheckedItems(["Normal Lifetime"])
    widget.mapping_export_controls["histogram"].setCheckedItems(["PNG", "CSV"])

    widget.run_batch()

    pngs = list(out_root.rglob("*_grouped_histogram.png"))
    assert pngs
    assert all(p.stat().st_size > 0 for p in pngs)


def test_collect_plot_settings(qtbot, make_viewer_model, tmp_path):
    (tmp_path / "a.ome.tif").write_bytes(b"x")
    widget = BatchAnalysisWidget(make_viewer_model())
    qtbot.addWidget(widget)
    widget._input_folder = str(tmp_path)
    widget._rescan()
    widget._export_folder = str(tmp_path)

    # A phasor plot alone makes a run valid even without a layer export.
    widget.export_ometiff_checkbox.setChecked(False)
    widget.plot_white_bg_checkbox.setChecked(True)
    widget.plot_colormap_combo.setCurrentText("viridis")
    _select_plot_type(widget.plot_type_combo, "Contour")
    widget.plot_center_checkbox.setChecked(True)
    widget._update_run_enabled()
    assert widget.run_button.isEnabled()

    plot_settings = widget._collect_plot_settings()
    assert plot_settings["white_background"] is True
    assert plot_settings["colormap"] == "viridis"
    assert plot_settings["plot_type"] == "Contour"
    assert plot_settings["show_center"] is True
    # The shared zoom settings ride along on the display dict.
    assert plot_settings["zoom"]["export"] is False


def test_store_plot_settings():
    layer = _make_phasor_layer()
    _store_plot_settings(
        layer,
        {
            "semi_circle": False,
            "log_scale": True,
            "white_background": True,
            "colormap": "viridis",
        },
    )
    settings = layer.metadata["settings"]
    assert settings["semi_circle"] is False
    assert settings["log_scale"] is True
    assert settings["white_background"] is True
    assert settings["colormap"] == "viridis"


def test_run_batch_plot_settings_and_centers(
    qtbot, make_viewer_model, tmp_path
):
    in_root = tmp_path / "in"
    in_root.mkdir()
    out_root = tmp_path / "out"
    out_root.mkdir()
    write_ome_tiff(
        str(in_root / "a.ome.tif"),
        _make_phasor_layer(name="a", harmonic=[1, 2]),
    )
    write_ome_tiff(
        str(in_root / "b.ome.tif"),
        _make_phasor_layer(name="b", harmonic=[1, 2]),
    )

    widget = BatchAnalysisWidget(make_viewer_model())
    qtbot.addWidget(widget)
    widget._input_folder = str(in_root)
    widget._rescan()
    idx = widget.format_combobox.findData(".ome.tif")
    widget.format_combobox.setCurrentIndex(idx)
    widget.harmonics_edit.setText("1, 2")
    widget._export_folder = str(out_root)
    widget.export_ometiff_checkbox.setChecked(True)
    widget.suffix_edit.setText("_analyzed")

    widget.plot_white_bg_checkbox.setChecked(True)
    widget.plot_colormap_combo.setCurrentText("viridis")
    widget.plot_centers_checkbox.setChecked(True)

    widget.run_batch()

    # Phasor centers CSV: header + 2 files x 2 harmonics.
    centers_csv = out_root / "Phasor Centers" / "phasor_centers.csv"
    assert centers_csv.exists()
    with open(centers_csv, newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["File", "Harmonic", "Center G", "Center S"]
    assert len(rows) == 1 + 2 * 2

    # One phasor-plot PNG per file per harmonic.
    plots = list(out_root.rglob("*_phasor_plot_H*.png"))
    assert len(plots) == 4

    # Plot settings persisted into the exported OME-TIFF.
    persisted = read_ome_tiff_settings(
        str(out_root / "OME-TIFF" / "a_analyzed.ome.tif")
    )
    assert persisted["white_background"] is True
    assert persisted["colormap"] == "viridis"


def test_run_batch_exports_phasor_plots(qtbot, make_viewer_model, tmp_path):
    in_root = tmp_path / "in"
    in_root.mkdir()
    out_root = tmp_path / "out"
    out_root.mkdir()
    write_ome_tiff(
        str(in_root / "a.ome.tif"),
        _make_phasor_layer(name="a", harmonic=[1, 2]),
    )

    widget = BatchAnalysisWidget(make_viewer_model())
    qtbot.addWidget(widget)
    widget._input_folder = str(in_root)
    widget._rescan()
    idx = widget.format_combobox.findData(".ome.tif")
    widget.format_combobox.setCurrentIndex(idx)
    widget.harmonics_edit.setText("1, 2")
    widget._export_folder = str(out_root)
    widget.export_ometiff_checkbox.setChecked(True)

    # Base phasor plot from Plot Settings (individual export on by default).
    # Components phasor plot (with component overlay).
    widget.components_group.setChecked(True)
    widget.components_plot_toggle.setChecked(True)

    widget.run_batch()

    plots = {p.name for p in out_root.rglob("*.png")}
    # One PNG per harmonic (data has harmonics 1 and 2).
    assert any("_phasor_plot_H1.png" in name for name in plots)
    assert any("_phasor_plot_H2.png" in name for name in plots)
    assert any("_components_phasor_H1.png" in name for name in plots)


# -- Round 4: colormaps, clustering, per-tab exports -----------------------


def test_mapping_colormap_and_contrast():
    layer = _make_phasor_layer(harmonic=[1, 2])
    pipeline = BatchPipeline(
        mapping={
            "output_type": "Normal Lifetime",
            "frequency": 80.0,
            "harmonic": 1,
            "colormap": "viridis",
            "contrast_limits": (0.0, 5.0),
        }
    )
    out = apply_pipeline(layer, pipeline)[0]
    assert out.colormap.name == "viridis"
    assert tuple(out.contrast_limits) == (0.0, 5.0)


def test_component_colormaps():
    layer = _make_phasor_layer(harmonic=[1, 2])
    pipeline = BatchPipeline(
        components={
            "analysis_type": "fit",
            "names": ["A", "B", "C"],
            "component_real": [0.1, 0.5, 0.8],
            "component_imag": [0.2, 0.4, 0.3],
            "colormaps": ["magma", "viridis", "turbo"],
            "contrast_limits": None,
            "harmonic": 1,
        }
    )
    layers = apply_pipeline(layer, pipeline)
    assert [lyr.colormap.name for lyr in layers] == [
        "magma",
        "viridis",
        "turbo",
    ]


def test_selection_cluster_pipeline():
    layer = _make_phasor_layer(harmonic=[1, 2])
    pipeline = BatchPipeline(
        selection={
            "harmonic": 1,
            "mode": "cluster",
            "cursors": [],
            "cluster": {"clusters": 2, "sigma": 2.0},
        }
    )
    extra_layers = apply_pipeline(layer, pipeline)
    assert len(extra_layers) == 1
    assert extra_layers[0].name.startswith("Cluster selection")
    assert extra_layers[0].data.shape == layer.data.shape


def test_collect_cursor_color(qtbot, make_viewer_model):
    widget = BatchAnalysisWidget(make_viewer_model())
    qtbot.addWidget(widget)
    widget.selection_group.setChecked(True)
    cursors = widget.build_pipeline([1, 2]).selection["cursors"]
    assert cursors[0]["color"].startswith("#")


def test_collect_selection_cluster_mode(qtbot, make_viewer_model):
    widget = BatchAnalysisWidget(make_viewer_model())
    qtbot.addWidget(widget)
    widget.selection_group.setChecked(True)
    idx = widget.selection_mode_combo.findData("cluster")
    widget.selection_mode_combo.setCurrentIndex(idx)
    widget.cluster_count_spin.setValue(3)
    selection = widget.build_pipeline([1, 2]).selection
    assert selection["mode"] == "cluster"
    assert selection["cluster"]["clusters"] == 3


def test_per_tab_exports_end_to_end(qtbot, make_viewer_model, tmp_path):
    in_root = tmp_path / "in"
    in_root.mkdir()
    out_root = tmp_path / "out"
    out_root.mkdir()
    write_ome_tiff(str(in_root / "a.ome.tif"), _make_phasor_layer(name="a"))

    widget = BatchAnalysisWidget(make_viewer_model())
    qtbot.addWidget(widget)
    widget._input_folder = str(in_root)
    widget._rescan()
    idx = widget.format_combobox.findData(".ome.tif")
    widget.format_combobox.setCurrentIndex(idx)
    widget._export_folder = str(out_root)
    widget.export_ometiff_checkbox.setChecked(False)

    widget.mapping_group.setChecked(True)
    widget.mapping_export_controls["stats"].setChecked(True)
    widget.mapping_export_controls["histogram"].setCheckedItems(["PNG", "CSV"])

    widget.run_batch()

    assert list(out_root.rglob("phasor_mapping_statistics.csv"))
    assert list(out_root.rglob("*_histogram.png"))
    assert list(out_root.rglob("*_histogram.csv"))


# -- Round 5: grouping / aggregate outputs ---------------------------------


def test_group_for_modes(qtbot, make_viewer_model):
    widget = BatchAnalysisWidget(make_viewer_model())
    qtbot.addWidget(widget)

    # Default Merged: everything is one group.
    assert (
        widget._group_for("a.ome.tif")[0] == widget._group_for("b.ome.tif")[0]
    )

    widget._group_config = {
        "mode": "Grouped",
        "assignments": {"a.ome.tif": 1, "b.ome.tif": 2},
        "group_names": {1: "Ctrl", 2: "Treated"},
        "group_colors": {1: "#ff0000", 2: "#0000ff"},
        "show_sd": False,
        "central_tendency": "None",
        "show_legend": True,
    }
    assert widget._group_for("a.ome.tif") == (1, "Ctrl", "#ff0000")
    assert widget._group_for("b.ome.tif") == (2, "Treated", "#0000ff")


def test_contour_key_styles_grouped(qtbot, make_viewer_model):
    """Grouped contour styles map each group id to its dialog style."""
    widget = BatchAnalysisWidget(make_viewer_model())
    qtbot.addWidget(widget)
    widget._group_config.update(
        {
            "mode": "Grouped",
            "contour_group_styles": {
                1: {"mode": "colormap", "colormap": "viridis"},
                2: {"mode": "solid", "color": (1.0, 0.0, 0.0)},
            },
        }
    )
    # items are (key, real, imag); the key is the group id in Grouped mode.
    items = [(1, [0.1], [0.2]), (2, [0.3], [0.4])]
    styles = widget._contour_key_styles(items)
    assert styles[1]["colormap"] == "viridis"
    assert styles[2]["mode"] == "solid"


def test_contour_key_styles_merged(qtbot, make_viewer_model):
    """Merged mode applies the single merged contour style to its key."""
    widget = BatchAnalysisWidget(make_viewer_model())
    qtbot.addWidget(widget)
    widget._group_config.update(
        {
            "mode": "Merged",
            "contour_merged_style": "colormap",
            "contour_merged_colormap": "magma",
        }
    )
    styles = widget._contour_key_styles([(0, [0.1], [0.2])])
    assert styles[0]["colormap"] == "magma"


def test_run_batch_grouped_outputs(qtbot, make_viewer_model, tmp_path):
    in_root = tmp_path / "in"
    in_root.mkdir()
    out_root = tmp_path / "out"
    out_root.mkdir()
    write_ome_tiff(
        str(in_root / "a.ome.tif"),
        _make_phasor_layer(name="a", harmonic=[1, 2]),
    )
    write_ome_tiff(
        str(in_root / "b.ome.tif"),
        _make_phasor_layer(name="b", harmonic=[1, 2]),
    )

    widget = BatchAnalysisWidget(make_viewer_model())
    qtbot.addWidget(widget)
    widget._input_folder = str(in_root)
    widget._rescan()
    idx = widget.format_combobox.findData(".ome.tif")
    widget.format_combobox.setCurrentIndex(idx)
    widget.harmonics_edit.setText("1, 2")
    widget._export_folder = str(out_root)
    widget.export_ometiff_checkbox.setChecked(False)

    widget._group_config = {
        "mode": "Grouped",
        "assignments": {"a.ome.tif": 1, "b.ome.tif": 2},
        "group_names": {1: "Ctrl", 2: "Treated"},
        "group_colors": {1: "#ff0000", 2: "#0000ff"},
        "show_sd": True,
        "central_tendency": "Mean",
        "show_legend": True,
    }
    _select_plot_type(widget._plot_combined_controls["type"], "Contour")
    widget.plot_center_checkbox.setChecked(True)
    widget.mapping_group.setChecked(True)
    widget.mapping_export_controls["histogram"].setCheckedItems(["PNG", "CSV"])
    widget.mapping_export_controls["stats"].setChecked(True)

    widget.run_batch()

    # Combined per-group contour, one per harmonic; the phasor center is drawn
    # onto the contour plot itself rather than a separate centers PNG.
    combined_dir = out_root / "Combined analysis" / "Phasor Plots"
    assert (combined_dir / "combined_contour_H1.png").exists()
    assert (combined_dir / "combined_contour_H2.png").exists()
    assert not (combined_dir / "combined_phasor_centers_H1.png").exists()
    # Grouped histogram + grouped statistics (one row per group).
    assert list(out_root.rglob("phasor_mapping_*_grouped_histogram.png"))
    grouped_csvs = list(
        out_root.rglob("phasor_mapping_grouped_statistics.csv")
    )
    assert grouped_csvs
    with open(grouped_csvs[0], newline="") as handle:
        rows = list(csv.reader(handle))
    groups = {row[1] for row in rows[1:]}
    assert groups == {"Ctrl", "Treated"}


def test_group_config_persisted_and_restored(qtbot, make_viewer_model):
    widget = BatchAnalysisWidget(make_viewer_model())
    qtbot.addWidget(widget)
    layer = _make_phasor_layer()
    _store_plot_settings(
        layer,
        {
            "semi_circle": True,
            "log_scale": False,
            "white_background": False,
            "colormap": "turbo",
        },
        {
            "mode": "Grouped",
            "assignments": {"a.ome.tif": 2},
            "group_names": {2: "Treated"},
            "group_colors": {2: "#0000ff"},
            "show_sd": True,
            "central_tendency": "Median",
            "show_legend": False,
        },
    )
    stored = layer.metadata["settings"]
    assert "batch_group_config" in stored

    widget._apply_settings_to_ui(stored)
    assert widget._group_config["mode"] == "Grouped"
    assert widget._group_config["group_names"][2] == "Treated"
    assert widget._group_config["central_tendency"] == "Median"


# -- Round 7: image-file masking ------------------------------------------


def test_apply_image_mask():
    layer = _make_phasor_layer(harmonic=[1, 2])
    mask = np.zeros(layer.data.shape, dtype=np.uint8)
    mask[0, :] = 255  # keep first row, exclude the rest

    _apply_image_mask(layer, mask, invert=False)

    assert not np.isnan(layer.data[0]).any()
    assert np.isnan(layer.data[1]).all()
    # G/S (harmonics, *spatial) are masked on every harmonic.
    assert np.isnan(layer.metadata["G"][:, 1, :]).all()
    assert not np.isnan(layer.metadata["G"][:, 0, :]).any()
    assert layer.metadata["mask_invert"] is False


def test_apply_image_mask_invert():
    layer = _make_phasor_layer(harmonic=[1, 2])
    mask = np.zeros(layer.data.shape, dtype=np.uint8)
    mask[0, :] = 1
    _apply_image_mask(layer, mask, invert=True)
    # Inverted: pixels inside the mask (row 0) are excluded instead.
    assert np.isnan(layer.data[0]).all()
    assert not np.isnan(layer.data[1]).any()


def test_apply_image_mask_shape_mismatch():
    layer = _make_phasor_layer(harmonic=[1, 2])
    bad = np.ones((3, 3), dtype=np.uint8)
    try:
        _apply_image_mask(layer, bad)
    except ValueError as exc:
        assert "shape" in str(exc)
    else:
        raise AssertionError("expected ValueError for shape mismatch")


def test_mask_pipeline():
    layer = _make_phasor_layer(harmonic=[1, 2])
    mask = np.zeros(layer.data.shape, dtype=np.uint8)
    mask[0, :] = 1
    pipeline = BatchPipeline(mask={"array": mask, "invert": False})
    apply_pipeline(layer, pipeline)
    assert np.isnan(layer.data[1]).all()


def test_mask_name_matching(qtbot, make_viewer_model, tmp_path):
    in_root = tmp_path / "in"
    in_root.mkdir()
    masks = tmp_path / "masks"
    masks.mkdir()
    for name in ("ABC", "XYZ"):
        write_ome_tiff(
            str(in_root / f"{name}.ome.tif"),
            _make_phasor_layer(name=name, harmonic=[1, 2]),
        )
    (masks / "ABC_mask.png").write_bytes(b"")
    (masks / "mask_XYZ.png").write_bytes(b"")
    (masks / "unrelated.png").write_bytes(b"")

    widget = BatchAnalysisWidget(make_viewer_model())
    qtbot.addWidget(widget)
    widget._input_folder = str(in_root)
    widget._rescan()
    idx = widget.format_combobox.findData(".ome.tif")
    widget.format_combobox.setCurrentIndex(idx)
    widget.masks_group.setChecked(True)
    widget._mask_folders = [str(masks)]
    widget._scan_mask_files()

    chosen = {
        os.path.basename(path): os.path.basename(row["combo"].currentData())
        for path, row in widget._mask_rows.items()
    }
    assert chosen["ABC.ome.tif"] == "ABC_mask.png"
    assert chosen["XYZ.ome.tif"] == "mask_XYZ.png"


def test_run_batch_with_masks(qtbot, make_viewer_model, tmp_path):
    import imageio.v3 as iio

    in_root = tmp_path / "in"
    in_root.mkdir()
    masks = tmp_path / "masks"
    masks.mkdir()
    out_root = tmp_path / "out"
    out_root.mkdir()

    layer = _make_phasor_layer(name="ABC", harmonic=[1, 2])
    write_ome_tiff(str(in_root / "ABC.ome.tif"), layer)
    mask = np.zeros(layer.data.shape, dtype=np.uint8)
    mask[0, :] = 255
    iio.imwrite(str(masks / "ABC_mask.png"), mask)

    widget = BatchAnalysisWidget(make_viewer_model())
    qtbot.addWidget(widget)
    widget._input_folder = str(in_root)
    widget._rescan()
    idx = widget.format_combobox.findData(".ome.tif")
    widget.format_combobox.setCurrentIndex(idx)
    widget.harmonics_edit.setText("1, 2")
    widget._export_folder = str(out_root)
    widget.export_ometiff_checkbox.setChecked(True)
    widget.suffix_edit.setText("_analyzed")
    widget.masks_group.setChecked(True)
    widget._mask_folders = [str(masks)]
    widget._scan_mask_files()

    widget.run_batch()

    out = out_root / "OME-TIFF" / "ABC_analyzed.ome.tif"
    assert out.exists()
    reader = napari_get_reader(str(out))
    result = reader(str(out))
    g_array = result[0][1]["metadata"]["G"]
    # Excluded row is NaN; kept row is finite.
    assert np.isnan(g_array[:, 1, :]).all()
    assert not np.isnan(g_array[:, 0, :]).any()


def test_derive_output_path_preserve_and_flatten(
    qtbot, make_viewer_model, tmp_path
):
    widget = BatchAnalysisWidget(make_viewer_model())
    qtbot.addWidget(widget)

    in_root = tmp_path / "in"
    out_root = tmp_path / "out"
    (in_root / "cond1").mkdir(parents=True)
    src = str(in_root / "cond1" / "sample.ome.tif")
    widget._input_folder = str(in_root)
    widget._export_folder = str(out_root)

    preserved = widget._derive_output_path(
        src, ".ome.tif", "_analyzed", preserve=True
    )
    assert preserved.endswith(os.path.join("cond1", "sample_analyzed"))

    flat = widget._derive_output_path(
        src, ".ome.tif", "_analyzed", preserve=False
    )
    assert os.path.dirname(flat) == str(out_root)
    assert flat.endswith("sample_analyzed")


# -- End to end ------------------------------------------------------------


def test_run_batch_end_to_end(qtbot, make_viewer_model, tmp_path):
    in_root = tmp_path / "in"
    sub = in_root / "cond1"
    sub.mkdir(parents=True)
    out_root = tmp_path / "out"
    out_root.mkdir()

    # Two phasor OME-TIFF inputs, one nested in a subfolder.
    layer_a = _make_phasor_layer(name="a")
    write_ome_tiff(str(in_root / "a.ome.tif"), layer_a)
    layer_b = _make_phasor_layer(name="b")
    write_ome_tiff(str(sub / "b.ome.tif"), layer_b)

    widget = BatchAnalysisWidget(make_viewer_model())
    qtbot.addWidget(widget)

    widget._input_folder = str(in_root)
    widget.subfolders_checkbox.setChecked(True)
    widget._rescan()
    # Select the OME-TIFF format.
    idx = widget.format_combobox.findData(".ome.tif")
    widget.format_combobox.setCurrentIndex(idx)

    widget._export_folder = str(out_root)
    widget.export_ometiff_checkbox.setChecked(True)
    widget.export_csv_checkbox.setChecked(True)
    widget.export_image_checkbox.setChecked(False)
    widget.suffix_edit.setText("_analyzed")

    widget.filter_group.setChecked(True)
    widget.filter_method_combo.setCurrentText("Median")
    widget.threshold_method_combo.setCurrentText("Manual")

    widget.run_batch()

    # Outputs preserve the subfolder tree and original names + suffix.
    out_a = out_root / "OME-TIFF" / "a_analyzed.ome.tif"
    out_b = out_root / "OME-TIFF" / "cond1" / "b_analyzed.ome.tif"
    assert out_a.exists()
    assert out_b.exists()
    assert (out_root / "CSV" / "a_analyzed.csv").exists()

    # Re-readable via the normal reader.
    reader = napari_get_reader(str(out_a))
    assert reader is not None
    result = reader(str(out_a))
    assert result and "G" in result[0][1]["metadata"]


def test_run_batch_exports_extra_layers(qtbot, make_viewer_model, tmp_path):
    in_root = tmp_path / "in"
    in_root.mkdir()
    out_root = tmp_path / "out"
    out_root.mkdir()
    write_ome_tiff(str(in_root / "a.ome.tif"), _make_phasor_layer(name="a"))

    widget = BatchAnalysisWidget(make_viewer_model())
    qtbot.addWidget(widget)
    widget._input_folder = str(in_root)
    widget._rescan()
    idx = widget.format_combobox.findData(".ome.tif")
    widget.format_combobox.setCurrentIndex(idx)

    widget._export_folder = str(out_root)
    widget.export_ometiff_checkbox.setChecked(True)
    widget.export_csv_checkbox.setChecked(False)
    widget.export_image_checkbox.setChecked(False)
    widget.suffix_edit.setText("_analyzed")

    # Mapping and selection both produce extra exported layers.
    widget.mapping_group.setChecked(True)
    widget.mapping_output_combo.setCheckedItems(["Normal Lifetime"])
    widget.selection_group.setChecked(True)

    widget.run_batch()

    # The colormapped mapping image is exported as PNG (per-tab default) into
    # its analysis subfolder; the main layer and selection labels are OME-TIFF.
    ome_files = [p.name for p in out_root.rglob("*.ome.tif")]
    assert any(name.startswith("a_analyzed") for name in ome_files)
    mapping_images = list(
        (
            out_root
            / "Phasor Mapping"
            / "Individual image analysis"
            / "Images"
        ).glob("*.png")
    )
    assert mapping_images


# -- Round 5: component overlay style + export modes -----------------------


def test_collect_components_includes_overlay_style(qtbot, make_viewer_model):
    """_collect_components carries line/label style and a colormap ramp."""
    widget = BatchAnalysisWidget(make_viewer_model())
    qtbot.addWidget(widget)
    widget.components_group.setChecked(True)
    # Customize the overlay style state (as the dialogs would).
    widget._component_line_style["line_width"] = 5.0
    widget._component_line_style["show_colormap_line"] = True
    widget._component_label_style["bold"] = True
    widget._component_rows[0]["colormap"].setCurrentText("viridis")

    components = widget.build_pipeline([1, 2]).components
    assert components["line_style"]["line_width"] == 5.0
    assert components["label_style"]["bold"] is True
    # The colormap ramp drives the colormap line / dot colors.
    assert components["fractions_colormap"] is not None
    assert len(components["fractions_colormap"]) > 1


def test_component_dot_colors_linear_uses_colormap_ends():
    """2-component Linear Projection dots take the colormap-end colors."""
    from napari_phasors._batch_analysis import (
        _colormap_color_list,
        _component_dot_colors,
    )

    ramp = _colormap_color_list("viridis")
    colors = _component_dot_colors(
        "Linear Projection",
        ["viridis", "viridis"],
        ramp,
        (0.0, 1.0),
        True,
        "dimgray",
    )
    assert len(colors) == 2
    # Component 1 maps to fraction 1.0 (ramp end), component 2 to 0.0 (start).
    assert colors[0] == ramp[-1]
    assert colors[1] == ramp[0]


def test_component_dot_colors_default_when_no_colormap_line():
    """With the overlay colormap off, all dots use the default color."""
    from napari_phasors._batch_analysis import _component_dot_colors

    colors = _component_dot_colors(
        "Linear Projection",
        ["viridis", "magma"],
        None,
        (0.0, 1.0),
        False,
        "dimgray",
    )
    assert colors == ["dimgray", "dimgray"]


def test_run_batch_components_plot_with_style(
    qtbot, make_viewer_model, tmp_path
):
    """End-to-end: a styled component overlay renders without error."""
    in_root = tmp_path / "in"
    in_root.mkdir()
    out_root = tmp_path / "out"
    out_root.mkdir()
    write_ome_tiff(
        str(in_root / "a.ome.tif"), _make_phasor_layer(name="a", harmonic=[1])
    )

    widget = BatchAnalysisWidget(make_viewer_model())
    qtbot.addWidget(widget)
    widget._input_folder = str(in_root)
    widget._rescan()
    idx = widget.format_combobox.findData(".ome.tif")
    widget.format_combobox.setCurrentIndex(idx)
    widget.harmonics_edit.setText("1")
    widget._export_folder = str(out_root)
    widget.export_ometiff_checkbox.setChecked(False)

    widget.components_group.setChecked(True)
    widget.components_plot_toggle.setChecked(True)
    # Colormap line on with a custom width.
    widget._component_line_style["show_colormap_line"] = True
    widget._component_line_style["line_width"] = 4.0
    widget._component_rows[0]["g"].setText("0.8")
    widget._component_rows[0]["s"].setText("0.3")
    widget._component_rows[1]["g"].setText("0.2")
    widget._component_rows[1]["s"].setText("0.3")

    widget.run_batch()

    plots = {p.name for p in out_root.rglob("*.png")}
    assert any("_components_phasor_H1.png" in name for name in plots)


def test_combined_merged_histogram_exported(
    qtbot, make_viewer_model, tmp_path
):
    """Histogram plot type with Combined on pools all files into one PNG."""
    in_root = tmp_path / "in"
    in_root.mkdir()
    out_root = tmp_path / "out"
    out_root.mkdir()
    for name in ("a", "b"):
        write_ome_tiff(
            str(in_root / f"{name}.ome.tif"),
            _make_phasor_layer(name=name, harmonic=[1]),
        )

    widget = BatchAnalysisWidget(make_viewer_model())
    qtbot.addWidget(widget)
    widget._input_folder = str(in_root)
    widget._rescan()
    idx = widget.format_combobox.findData(".ome.tif")
    widget.format_combobox.setCurrentIndex(idx)
    widget.harmonics_edit.setText("1")
    widget._export_folder = str(out_root)
    widget.export_ometiff_checkbox.setChecked(False)

    _select_plot_type(widget._plot_combined_controls["type"], "Histogram")
    widget.plot_combined_checkbox.setChecked(True)
    widget._group_config["mode"] = "Merged"

    widget.run_batch()

    assert (
        out_root
        / "Combined analysis"
        / "Phasor Plots"
        / "combined_phasor_H1.png"
    ).exists()


def test_individual_checkbox_gates_per_file_plots(
    qtbot, make_viewer_model, tmp_path
):
    """Unchecking Individual suppresses the per-file base phasor plots."""
    in_root = tmp_path / "in"
    in_root.mkdir()
    out_root = tmp_path / "out"
    out_root.mkdir()
    write_ome_tiff(
        str(in_root / "a.ome.tif"), _make_phasor_layer(name="a", harmonic=[1])
    )

    widget = BatchAnalysisWidget(make_viewer_model())
    qtbot.addWidget(widget)
    widget._input_folder = str(in_root)
    widget._rescan()
    idx = widget.format_combobox.findData(".ome.tif")
    widget.format_combobox.setCurrentIndex(idx)
    widget.harmonics_edit.setText("1")
    widget._export_folder = str(out_root)
    widget.export_ometiff_checkbox.setChecked(False)

    _select_plot_type(widget._plot_combined_controls["type"], "Histogram")
    widget.plot_individual_checkbox.setChecked(False)
    widget.plot_combined_checkbox.setChecked(True)

    widget.run_batch()

    per_file = list(out_root.rglob("*_phasor_plot_H*.png"))
    assert per_file == []
    # Combined output is still produced.
    assert (
        out_root
        / "Combined analysis"
        / "Phasor Plots"
        / "combined_phasor_H1.png"
    ).exists()


def test_combined_contour_grouped_with_styles(
    qtbot, make_viewer_model, tmp_path
):
    """Contour + Grouped + per-group styles renders a combined contour PNG."""
    in_root = tmp_path / "in"
    in_root.mkdir()
    out_root = tmp_path / "out"
    out_root.mkdir()
    for name in ("a", "b"):
        write_ome_tiff(
            str(in_root / f"{name}.ome.tif"),
            _make_phasor_layer(name=name, harmonic=[1]),
        )

    widget = BatchAnalysisWidget(make_viewer_model())
    qtbot.addWidget(widget)
    widget._input_folder = str(in_root)
    widget._rescan()
    idx = widget.format_combobox.findData(".ome.tif")
    widget.format_combobox.setCurrentIndex(idx)
    widget.harmonics_edit.setText("1")
    widget._export_folder = str(out_root)
    widget.export_ometiff_checkbox.setChecked(False)

    _select_plot_type(widget._plot_combined_controls["type"], "Contour")
    widget.plot_combined_checkbox.setChecked(True)
    widget._group_config.update(
        {
            "mode": "Grouped",
            "assignments": {"a.ome.tif": 1, "b.ome.tif": 2},
            "group_names": {1: "A", 2: "B"},
            "contour_group_styles": {
                1: {"mode": "colormap", "colormap": "viridis"},
                2: {"mode": "solid", "color": (1.0, 0.0, 0.0)},
            },
        }
    )

    widget.run_batch()

    assert (
        out_root
        / "Combined analysis"
        / "Phasor Plots"
        / "combined_contour_H1.png"
    ).exists()


def test_open_plot_group_dialog_round_trips_groups(
    qtbot, make_viewer_model, tmp_path, monkeypatch
):
    """The Configure Groups dialog persists assignments + legend into config."""
    from qtpy.QtWidgets import QDialog

    import napari_phasors.plotter as plotter_mod

    write_ome_tiff(str(tmp_path / "a.ome.tif"), _make_phasor_layer(name="a"))
    write_ome_tiff(str(tmp_path / "b.ome.tif"), _make_phasor_layer(name="b"))

    widget = BatchAnalysisWidget(make_viewer_model())
    qtbot.addWidget(widget)
    widget._input_folder = str(tmp_path)
    widget._rescan()
    idx = widget.format_combobox.findData(".ome.tif")
    widget.format_combobox.setCurrentIndex(idx)
    # Two groups must be assigned before opening, since the groups-only dialog
    # builds its rows from the existing assignments.
    widget._group_config["assignments"] = {
        "a.ome.tif": 1,
        "b.ome.tif": 2,
    }

    # Drive the dialog non-interactively: it opens locked to Grouped mode.
    def fake_exec(self):
        self._show_legend_checkbox.setChecked(False)
        return QDialog.Accepted

    monkeypatch.setattr(
        plotter_mod.ContourLayerSettingsDialog, "exec", fake_exec
    )
    widget._open_plot_group_dialog()

    assert widget._group_config["mode"] == "Grouped"
    assert set(widget._group_config["assignments"].values()) == {1, 2}
    assert widget._group_config["show_legend"] is False
    # Per-group contour styling is captured for the renderer.
    assert "contour_group_styles" in widget._group_config


# -- Round 6: clean export names + per-tab combined phasor plots -----------


def test_analysis_histogram_filename_no_duplication(
    qtbot, make_viewer_model, tmp_path
):
    """Histogram files are named <stem>_<label>_histogram with no repetition."""
    in_root = tmp_path / "in"
    in_root.mkdir()
    out_root = tmp_path / "out"
    out_root.mkdir()
    write_ome_tiff(
        str(in_root / "sample one.ome.tif"),
        _make_phasor_layer(name="sample one", harmonic=[1]),
    )

    widget = BatchAnalysisWidget(make_viewer_model())
    qtbot.addWidget(widget)
    widget._input_folder = str(in_root)
    widget._rescan()
    idx = widget.format_combobox.findData(".ome.tif")
    widget.format_combobox.setCurrentIndex(idx)
    widget.harmonics_edit.setText("1")
    widget._export_folder = str(out_root)
    widget.suffix_edit.setText("")
    widget.export_ometiff_checkbox.setChecked(False)

    widget.components_group.setChecked(True)
    widget._component_rows[0]["g"].setText("0.8")
    widget._component_rows[0]["s"].setText("0.3")
    widget._component_rows[1]["g"].setText("0.2")
    widget._component_rows[1]["s"].setText("0.3")
    widget.components_export_controls["histogram"].setCheckedItems(
        ["PNG", "CSV"]
    )

    widget.run_batch()

    hist_csvs = list(out_root.rglob("*_histogram.csv"))
    names = {p.name for p in hist_csvs}
    # Clean: source stem (kept verbatim) once, the safe-ified analysis label,
    # then _histogram. The label's spaces become underscores; the file stem is
    # left untouched. Directory iteration order isn't guaranteed across
    # filesystems, so check set membership rather than indexing into the
    # glob result.
    assert "sample one_Component_1_fraction_histogram.csv" in names
    # No leftover layer-name duplication.
    for name in names:
        assert "Intensity" not in name
        assert name.count("sample one") == 1


def test_combined_components_phasor_merged(qtbot, make_viewer_model, tmp_path):
    """Components phasor plot can be exported combined (merged) across files."""
    in_root = tmp_path / "in"
    in_root.mkdir()
    out_root = tmp_path / "out"
    out_root.mkdir()
    for name in ("a", "b"):
        write_ome_tiff(
            str(in_root / f"{name}.ome.tif"),
            _make_phasor_layer(name=name, harmonic=[1]),
        )

    widget = BatchAnalysisWidget(make_viewer_model())
    qtbot.addWidget(widget)
    widget._input_folder = str(in_root)
    widget._rescan()
    idx = widget.format_combobox.findData(".ome.tif")
    widget.format_combobox.setCurrentIndex(idx)
    widget.harmonics_edit.setText("1")
    widget._export_folder = str(out_root)
    widget.export_ometiff_checkbox.setChecked(False)

    # Components phasor plot enabled; base Plot Settings tab left disabled.
    widget.components_group.setChecked(True)
    widget.components_plot_toggle.setChecked(True)
    widget.plot_individual_checkbox.setChecked(True)
    widget.plot_combined_checkbox.setChecked(True)
    widget._group_config["mode"] = "Merged"

    widget.run_batch()

    combined = list(out_root.rglob("combined_components_phasor_H1.png"))
    assert combined, "expected a merged combined components phasor plot"
    # Individual per-file plots are also produced (exclude the combined one).
    per_file = [
        p
        for p in out_root.rglob("*_components_phasor_H1.png")
        if not p.name.startswith("combined_")
    ]
    assert {p.name for p in per_file} == {
        "a_components_phasor_H1.png",
        "b_components_phasor_H1.png",
    }


def test_combined_components_phasor_grouped(
    qtbot, make_viewer_model, tmp_path
):
    """Grouped mode yields one combined components plot per group."""
    in_root = tmp_path / "in"
    in_root.mkdir()
    out_root = tmp_path / "out"
    out_root.mkdir()
    for name in ("a", "b", "c"):
        write_ome_tiff(
            str(in_root / f"{name}.ome.tif"),
            _make_phasor_layer(name=name, harmonic=[1]),
        )

    widget = BatchAnalysisWidget(make_viewer_model())
    qtbot.addWidget(widget)
    widget._input_folder = str(in_root)
    widget._rescan()
    idx = widget.format_combobox.findData(".ome.tif")
    widget.format_combobox.setCurrentIndex(idx)
    widget.harmonics_edit.setText("1")
    widget._export_folder = str(out_root)
    widget.export_ometiff_checkbox.setChecked(False)

    widget.components_group.setChecked(True)
    widget.components_plot_toggle.setChecked(True)
    widget.plot_individual_checkbox.setChecked(False)  # combined only
    widget.plot_combined_checkbox.setChecked(True)
    widget._group_config.update(
        {
            "mode": "Grouped",
            "assignments": {
                "a.ome.tif": 1,
                "b.ome.tif": 1,
                "c.ome.tif": 2,
            },
            "group_names": {1: "GroupA", 2: "GroupB"},
        }
    )

    widget.run_batch()

    files = {
        p.name for p in out_root.rglob("combined_components_phasor_*.png")
    }
    assert any("GroupA" in name for name in files)
    assert any("GroupB" in name for name in files)
    # Individual was off, so no per-file component plots.
    assert not list(out_root.rglob("a_components_phasor_H1.png"))


def test_apply_selection_settings_populates_cursors(qtbot, make_viewer_model):
    """Manual cursors stored by the Selection tab are loaded into the UI."""
    widget = BatchAnalysisWidget(make_viewer_model())
    qtbot.addWidget(widget)

    # Structure exactly as the interactive Selection tab persists it under
    # metadata["settings"]["selections"].
    settings = {
        "selections": {
            "circular_cursors": [
                {"g": 0.4, "s": 0.3, "radius": 0.05, "color": [255, 0, 0, 255]}
            ],
            "elliptical_cursors": [
                {
                    "g": 0.6,
                    "s": 0.2,
                    "radius": 0.08,
                    "radius_minor": 0.04,
                    "angle": 30.0,
                    "color": [0, 255, 0, 255],
                }
            ],
            "polar_cursors": [
                {
                    "phase_min": 0.1,
                    "phase_max": 0.9,
                    "modulation_min": 0.2,
                    "modulation_max": 0.8,
                    "color": [0, 0, 255, 255],
                }
            ],
        }
    }

    widget._apply_settings_to_ui(settings)

    assert widget.selection_group.isChecked()
    assert widget.selection_mode_combo.currentData() == "manual"
    rows = widget._cursor_rows
    assert len(rows) == 3

    circ, elli, pol = rows
    assert circ["type"].currentData() == "circular"
    assert float(circ["g"].text()) == 0.4
    assert float(circ["s"].text()) == 0.3
    assert circ["radius"].value() == 0.05
    assert circ["color"].color().getRgb() == (255, 0, 0, 255)

    assert elli["type"].currentData() == "elliptic"
    assert elli["radius_minor"].value() == 0.04
    assert elli["angle"].value() == 30.0

    assert pol["type"].currentData() == "polar"
    assert pol["phase_min"].value() == 0.1
    assert pol["mod_max"].value() == 0.8

    # The loaded cursors collect back into a runnable selection pipeline.
    selection = widget._collect_selection()
    types = [c["type"] for c in selection["cursors"]]
    assert types == ["circular", "elliptic", "polar"]


def test_collect_plot_settings_individual_vs_combined(
    qtbot, make_viewer_model
):
    """Individual and combined plots are styled independently."""
    widget = BatchAnalysisWidget(make_viewer_model())
    qtbot.addWidget(widget)

    _select_plot_type(widget._plot_individual_controls["type"], "Histogram")
    widget._plot_individual_controls["bins"].setValue(123)
    _select_plot_type(widget._plot_combined_controls["type"], "Contour")
    widget._plot_combined_controls["levels"].setValue(11)

    individual = widget._collect_plot_settings("individual")
    combined = widget._collect_plot_settings("combined")

    assert individual["plot_type"] == "Histogram"
    assert individual["bins"] == 123
    assert combined["plot_type"] == "Contour"
    assert combined["contour_levels"] == 11
    # The default (no-arg) collection is the individual mode.
    assert widget._collect_plot_settings()["plot_type"] == "Histogram"


def test_collect_zoom_settings(qtbot, make_viewer_model):
    """The zoom controls produce a shared zoom dict on both modes."""
    widget = BatchAnalysisWidget(make_viewer_model())
    qtbot.addWidget(widget)

    widget.plot_zoom_checkbox.setChecked(True)
    widget.plot_zoom_rect_checkbox.setChecked(True)
    widget.plot_zoom_xmin.setValue(0.1)
    widget.plot_zoom_xmax.setValue(0.7)
    widget.plot_zoom_ymin.setValue(0.05)
    widget.plot_zoom_ymax.setValue(0.5)

    zoom = widget._collect_zoom_settings()
    assert zoom["export"] is True
    assert zoom["rectangle"] is True
    assert zoom["xmin"] == 0.1
    assert zoom["xmax"] == 0.7
    assert zoom["ymin"] == 0.05
    assert zoom["ymax"] == 0.5
    # The same zoom dict rides along on the collected display settings.
    assert widget._collect_plot_settings("combined")["zoom"]["export"] is True


def test_run_batch_exports_zoomed_section(qtbot, make_viewer_model, tmp_path):
    """Enabling zoom writes a '_zoom' PNG alongside each full phasor plot."""
    in_root = tmp_path / "in"
    in_root.mkdir()
    out_root = tmp_path / "out"
    out_root.mkdir()
    write_ome_tiff(
        str(in_root / "a.ome.tif"),
        _make_phasor_layer(name="a", harmonic=[1]),
    )

    widget = BatchAnalysisWidget(make_viewer_model())
    qtbot.addWidget(widget)
    widget._input_folder = str(in_root)
    widget._rescan()
    idx = widget.format_combobox.findData(".ome.tif")
    widget.format_combobox.setCurrentIndex(idx)
    widget.harmonics_edit.setText("1")
    widget._export_folder = str(out_root)
    widget.export_ometiff_checkbox.setChecked(False)

    # Individual only, with a zoomed crop and the rectangle on the full plot.
    widget.plot_individual_checkbox.setChecked(True)
    widget.plot_combined_checkbox.setChecked(False)
    widget.plot_zoom_checkbox.setChecked(True)
    widget.plot_zoom_rect_checkbox.setChecked(True)
    widget.plot_zoom_xmin.setValue(0.2)
    widget.plot_zoom_xmax.setValue(0.8)
    widget.plot_zoom_ymin.setValue(0.1)
    widget.plot_zoom_ymax.setValue(0.5)

    widget.run_batch()

    individual_dir = out_root / "Individual image analysis" / "Phasor Plots"
    full = individual_dir / "a_phasor_plot_H1.png"
    zoom = individual_dir / "a_phasor_plot_H1_zoom.png"
    assert full.exists()
    assert zoom.exists()


# -- Auto ranges / global contrast / subfolder structure -------------------


def test_auto_mapping_ranges_uses_all_files(
    qtbot, make_viewer_model, tmp_path
):
    """The mapping Auto button pools every file's phasor data for the range."""
    in_root = tmp_path / "in"
    in_root.mkdir()
    for name in ("a", "b"):
        write_ome_tiff(
            str(in_root / f"{name}.ome.tif"),
            _make_phasor_layer(name=name, harmonic=[1]),
        )

    widget = BatchAnalysisWidget(make_viewer_model())
    qtbot.addWidget(widget)
    widget._input_folder = str(in_root)
    widget._rescan()
    idx = widget.format_combobox.findData(".ome.tif")
    widget.format_combobox.setCurrentIndex(idx)
    widget.harmonics_edit.setText("1")
    widget.mapping_harmonic_spin.setValue(1)

    coords = widget._gather_all_phasor_coords(1)
    assert coords is not None
    g, s = coords
    assert g.size > 0 and s.size > 0

    # The Auto button must not raise (regression: it used self.layer_combo).
    widget._auto_mapping_ranges()
    assert (
        widget.mapping_phase_max_spin.value()
        >= widget.mapping_phase_min_spin.value()
    )
    assert (
        widget.mapping_mod_max_spin.value()
        >= widget.mapping_mod_min_spin.value()
    )


def test_global_contrast_shared_across_files(
    qtbot, make_viewer_model, tmp_path
):
    """Auto contrast pools a single range applied to every exported image."""
    in_root = tmp_path / "in"
    in_root.mkdir()
    out_root = tmp_path / "out"
    out_root.mkdir()
    for name in ("a", "b"):
        write_ome_tiff(
            str(in_root / f"{name}.ome.tif"),
            _make_phasor_layer(name=name, harmonic=[1]),
        )

    widget = BatchAnalysisWidget(make_viewer_model())
    qtbot.addWidget(widget)
    widget._input_folder = str(in_root)
    widget._rescan()
    idx = widget.format_combobox.findData(".ome.tif")
    widget.format_combobox.setCurrentIndex(idx)
    widget.harmonics_edit.setText("1")
    widget._export_folder = str(out_root)
    widget.export_ometiff_checkbox.setChecked(False)
    widget.export_image_checkbox.setChecked(True)

    widget.mapping_group.setChecked(True)
    widget.mapping_output_combo.setCheckedItems(["Phase"])
    widget.mapping_harmonic_spin.setValue(1)
    # Auto contrast is the default; ensure it.
    assert widget.mapping_contrast["auto"].isChecked()

    widget.run_batch()

    # A shared range was computed for the Phase output and is non-degenerate.
    assert "Phase" in widget._global_contrast
    lo, hi = widget._global_contrast["Phase"]
    assert hi > lo


def test_export_subfolder_structure(qtbot, make_viewer_model, tmp_path):
    """Exports are organized into typed subfolders per output and tab."""
    in_root = tmp_path / "in"
    in_root.mkdir()
    out_root = tmp_path / "out"
    out_root.mkdir()
    for name in ("a", "b"):
        write_ome_tiff(
            str(in_root / f"{name}.ome.tif"),
            _make_phasor_layer(name=name, harmonic=[1]),
        )

    widget = BatchAnalysisWidget(make_viewer_model())
    qtbot.addWidget(widget)
    widget._input_folder = str(in_root)
    widget._rescan()
    idx = widget.format_combobox.findData(".ome.tif")
    widget.format_combobox.setCurrentIndex(idx)
    widget.harmonics_edit.setText("1")
    widget._export_folder = str(out_root)
    widget.export_ometiff_checkbox.setChecked(True)
    widget.export_csv_checkbox.setChecked(True)
    widget.export_image_checkbox.setChecked(True)

    widget.mapping_group.setChecked(True)
    widget.mapping_output_combo.setCheckedItems(["Normal Lifetime"])
    widget.mapping_frequency_spin.setText("80")
    widget.mapping_harmonic_spin.setValue(1)
    widget.mapping_export_controls["histogram"].setCheckedItems(["PNG", "CSV"])
    widget.mapping_export_controls["stats"].setChecked(True)

    widget.plot_individual_checkbox.setChecked(True)
    widget.plot_combined_checkbox.setChecked(True)
    widget.plot_centers_checkbox.setChecked(True)

    widget.run_batch()

    expected = [
        out_root / "OME-TIFF",
        out_root / "CSV",
        out_root / "Intensity Images",
        out_root / "Individual image analysis" / "Phasor Plots",
        out_root / "Combined analysis" / "Phasor Plots",
        out_root / "Phasor Centers" / "phasor_centers.csv",
        out_root / "Phasor Mapping" / "Combined analysis" / "Statistics",
        out_root / "Phasor Mapping" / "Individual image analysis" / "Images",
        out_root
        / "Phasor Mapping"
        / "Individual image analysis"
        / "Histograms"
        / "PNG",
        out_root
        / "Phasor Mapping"
        / "Individual image analysis"
        / "Histograms"
        / "CSV",
    ]
    for path in expected:
        assert path.exists(), f"missing {path}"


def test_mapping_range_auto_checkbox_disables_fields(qtbot, make_viewer_model):
    """The mesh-range Auto checkbox disables the manual range spinboxes."""
    widget = BatchAnalysisWidget(make_viewer_model())
    qtbot.addWidget(widget)
    # Enable the tab so the section body (and its fields) are interactive.
    widget.mapping_group.setChecked(True)

    # Auto is on by default -> manual fields disabled.
    assert widget.mapping_range_auto_checkbox.isChecked()
    assert not widget.mapping_phase_min_spin.isEnabled()
    assert not widget.mapping_mod_max_spin.isEnabled()

    widget.mapping_range_auto_checkbox.setChecked(False)
    assert widget.mapping_phase_min_spin.isEnabled()
    assert widget.mapping_mod_max_spin.isEnabled()


def test_analysis_runs_once_per_file(
    qtbot, make_viewer_model, tmp_path, monkeypatch
):
    """The component analysis is computed once per file (no double pass)."""
    import napari_phasors._batch_analysis as batch_mod

    in_root = tmp_path / "in"
    in_root.mkdir()
    out_root = tmp_path / "out"
    out_root.mkdir()
    for name in ("a", "b"):
        write_ome_tiff(
            str(in_root / f"{name}.ome.tif"),
            _make_phasor_layer(name=name, harmonic=[1]),
        )

    calls = {"n": 0}
    original = batch_mod._apply_component_fraction

    def counting(layer, components):
        calls["n"] += 1
        return original(layer, components)

    monkeypatch.setattr(batch_mod, "_apply_component_fraction", counting)

    widget = BatchAnalysisWidget(make_viewer_model())
    qtbot.addWidget(widget)
    widget._input_folder = str(in_root)
    widget._rescan()
    idx = widget.format_combobox.findData(".ome.tif")
    widget.format_combobox.setCurrentIndex(idx)
    widget.harmonics_edit.setText("1")
    widget._export_folder = str(out_root)
    widget.export_ometiff_checkbox.setChecked(False)
    widget.export_image_checkbox.setChecked(True)

    widget.components_group.setChecked(True)
    widget._component_rows[0]["g"].setText("0.8")
    widget._component_rows[0]["s"].setText("0.3")
    widget._component_rows[1]["g"].setText("0.2")
    widget._component_rows[1]["s"].setText("0.3")
    widget.components_export_controls["histogram"].setCheckedItems(
        ["PNG", "CSV"]
    )
    widget.components_export_controls["stats"].setChecked(True)

    widget.run_batch()

    # Exactly one analysis call per file (via apply_pipeline), not 2x/3x.
    assert calls["n"] == 2
    # Deferred per-file fraction images were written for both files.
    images = list(
        (
            out_root
            / "Component Analysis"
            / "Individual image analysis"
            / "Images"
        ).glob("*.png")
    )
    assert len(images) >= 2
    # The shared range was computed once and applied.
    assert widget._global_contrast
    # The deferral temp dir is cleaned up.
    assert widget._deferred_dir is None or not os.path.exists(
        widget._deferred_dir
    )


def test_histogram_csv_respects_value_range(tmp_path):
    """A value_range bounds the histogram CSV to the chosen range limits."""
    from napari_phasors._batch_analysis import _save_histogram_csv

    values = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    path = tmp_path / "h.csv"
    _save_histogram_csv(values, 4, str(path), value_range=(1.0, 3.0))

    with open(path, newline="") as handle:
        rows = list(csv.reader(handle))
    centers = [float(r[0]) for r in rows[1:]]
    counts = [int(r[1]) for r in rows[1:]]
    # All bin centers fall inside the requested range...
    assert min(centers) >= 1.0 and max(centers) <= 3.0
    # ...and only the three in-range values (1, 2, 3) are counted.
    assert sum(counts) == 3


def test_output_controls_unified(qtbot, make_viewer_model):
    """Each analysis tab exposes one Outputs section with format comboboxes."""
    widget = BatchAnalysisWidget(make_viewer_model())
    qtbot.addWidget(widget)

    controls = widget.mapping_export_controls
    # Image + histogram formats are checkable comboboxes; image defaults PNG.
    assert controls["image"].checkedItems() == ["PNG"]
    assert controls["histogram"].checkedItems() == []
    # The phasor-plot toggle lives inside the unified section.
    assert widget.mapping_plot_toggle is controls["plot"]


def test_per_tab_image_format_csv_only(qtbot, make_viewer_model, tmp_path):
    """Selecting CSV-only exports the analysis image as CSV, not PNG."""
    in_root = tmp_path / "in"
    in_root.mkdir()
    out_root = tmp_path / "out"
    out_root.mkdir()
    write_ome_tiff(
        str(in_root / "a.ome.tif"), _make_phasor_layer(name="a", harmonic=[1])
    )

    widget = BatchAnalysisWidget(make_viewer_model())
    qtbot.addWidget(widget)
    widget._input_folder = str(in_root)
    widget._rescan()
    idx = widget.format_combobox.findData(".ome.tif")
    widget.format_combobox.setCurrentIndex(idx)
    widget.harmonics_edit.setText("1")
    widget._export_folder = str(out_root)
    widget.export_ometiff_checkbox.setChecked(False)

    widget.mapping_group.setChecked(True)
    widget.mapping_output_combo.setCheckedItems(["Phase"])
    widget.mapping_harmonic_spin.setValue(1)
    widget.mapping_export_controls["image"].setCheckedItems(["CSV"])

    widget.run_batch()

    img_dir = out_root / "Phasor Mapping" / "Individual image analysis"
    assert list((img_dir / "CSV").glob("*.csv"))
    # No colormapped PNG image was written.
    assert not (img_dir / "Images").exists()


def test_format_combobox_display_text(qtbot, make_viewer_model):
    """The PNG/CSV format combobox lists the selected formats literally."""
    widget = BatchAnalysisWidget(make_viewer_model())
    qtbot.addWidget(widget)
    combo = widget.mapping_export_controls["image"]

    combo.setCheckedItems(["PNG", "CSV"])
    assert combo.lineEdit().text() == "PNG, CSV"
    combo.setCheckedItems(["CSV"])
    assert combo.lineEdit().text() == "CSV"
    combo.setCheckedItems([])
    assert combo.lineEdit().text() == "None"


def test_groups_only_dialog_hides_mode(qtbot):
    """The groups-only contour dialog hides the mode selector (Grouped)."""
    from napari_phasors.plotter import ContourLayerSettingsDialog

    dialog = ContourLayerSettingsDialog(
        groups_only=True,
        layer_labels=["a", "b"],
        group_assignments={"a": 1, "b": 2},
    )
    qtbot.addWidget(dialog)
    assert dialog.mode_combo.currentText() == "Grouped"
    # The mode selector and its label are hidden in groups-only mode.
    assert not dialog.mode_combo.isVisibleTo(dialog)
    assert not dialog._mode_label.isVisibleTo(dialog)
    # The grouped section (colored checkable-combobox rows) is shown.
    assert dialog._group_section.isVisibleTo(dialog)


def test_linear_projection_exports_both_fractions():
    """Linear Projection yields both component fractions (second = 1 - first)."""
    layer = _make_phasor_layer(harmonic=[1])
    config = {
        "analysis_type": "linear",
        "names": ["A", "B"],
        "colormaps": ["turbo", "viridis"],
        "component_real": [0.8, 0.2],
        "component_imag": [0.3, 0.3],
        "harmonic": 1,
        "contrast_limits": None,
    }
    outputs = _apply_component_fraction(layer, config)
    assert len(outputs) == 2
    assert [o.name.split(":")[0] for o in outputs] == [
        "A fraction",
        "B fraction",
    ]
    first = np.asarray(outputs[0].data)
    second = np.asarray(outputs[1].data)
    finite = np.isfinite(first) & np.isfinite(second)
    np.testing.assert_allclose(second[finite], 1.0 - first[finite])


def test_combined_all_groups_plot_exported(qtbot, make_viewer_model, tmp_path):
    """Grouped combined export writes per-group AND an all-groups plot."""
    in_root = tmp_path / "in"
    in_root.mkdir()
    out_root = tmp_path / "out"
    out_root.mkdir()
    for name in ("a", "b"):
        write_ome_tiff(
            str(in_root / f"{name}.ome.tif"),
            _make_phasor_layer(name=name, harmonic=[1]),
        )

    widget = BatchAnalysisWidget(make_viewer_model())
    qtbot.addWidget(widget)
    widget._input_folder = str(in_root)
    widget._rescan()
    idx = widget.format_combobox.findData(".ome.tif")
    widget.format_combobox.setCurrentIndex(idx)
    widget.harmonics_edit.setText("1")
    widget._export_folder = str(out_root)
    widget.export_ometiff_checkbox.setChecked(False)

    _select_plot_type(widget._plot_combined_controls["type"], "Histogram")
    widget.plot_combined_checkbox.setChecked(True)
    widget._group_config.update(
        {
            "mode": "Grouped",
            "assignments": {"a.ome.tif": 1, "b.ome.tif": 2},
            "group_names": {1: "G1", 2: "G2"},
        }
    )

    widget.run_batch()

    combined = out_root / "Combined analysis" / "Phasor Plots"
    assert (combined / "combined_phasor_all_groups_H1.png").exists()
    # Per-group plots are still exported alongside the all-groups plot.
    per_group = [
        p
        for p in combined.glob("combined_phasor_*_H1.png")
        if "all_groups" not in p.name
    ]
    assert per_group


def test_combined_colormap_hidden_for_grouped_contour(
    qtbot, make_viewer_model
):
    """The combined colormap is hidden for a grouped contour (per-group cmap)."""
    widget = BatchAnalysisWidget(make_viewer_model())
    qtbot.addWidget(widget)
    cmap = widget._plot_combined_controls["colormap"]

    _select_plot_type(widget._plot_combined_controls["type"], "Contour")
    widget.plot_combined_mode_combo.setCurrentText("Merged")
    assert not cmap.isHidden()  # merged contour uses the combined colormap

    widget.plot_combined_mode_combo.setCurrentText("Grouped")
    assert cmap.isHidden()  # grouped: per-group colors instead

    # Scatter never shows the colormap regardless of grouping.
    _select_plot_type(widget._plot_combined_controls["type"], "Scatter")
    assert cmap.isHidden()


def test_tab_combined_contour_all_groups_exported(
    qtbot, make_viewer_model, tmp_path
):
    """Analysis-overlay combined contour exports per-group AND all-groups."""
    in_root = tmp_path / "in"
    in_root.mkdir()
    out_root = tmp_path / "out"
    out_root.mkdir()
    for name in ("a", "b"):
        write_ome_tiff(
            str(in_root / f"{name}.ome.tif"),
            _make_phasor_layer(name=name, harmonic=[1]),
        )

    widget = BatchAnalysisWidget(make_viewer_model())
    qtbot.addWidget(widget)
    widget._input_folder = str(in_root)
    widget._rescan()
    idx = widget.format_combobox.findData(".ome.tif")
    widget.format_combobox.setCurrentIndex(idx)
    widget.harmonics_edit.setText("1")
    widget._export_folder = str(out_root)
    widget.export_ometiff_checkbox.setChecked(False)

    widget.mapping_group.setChecked(True)
    widget.mapping_output_combo.setCheckedItems(["Phase"])
    widget.mapping_harmonic_spin.setValue(1)
    widget.mapping_plot_toggle.setChecked(True)

    widget.plot_combined_checkbox.setChecked(True)
    _select_plot_type(widget._plot_combined_controls["type"], "Contour")
    widget._group_config.update(
        {
            "mode": "Grouped",
            "assignments": {"a.ome.tif": 1, "b.ome.tif": 2},
            "group_names": {1: "G1", 2: "G2"},
        }
    )

    widget.run_batch()

    combined = (
        out_root / "Phasor Mapping" / "Combined analysis" / "Phasor Plots"
    )
    assert (combined / "combined_mapping_phasor_all_groups_H1.png").exists()
    per_group = [
        p
        for p in combined.glob("combined_mapping_phasor_*_H1.png")
        if "all_groups" not in p.name
    ]
    assert per_group


def test_streaming_histogram_keeps_per_file_for_sd(tmp_path):
    """Streaming mode reloads per-file arrays so the SD band can be drawn."""
    from napari_phasors._batch_analysis import _SpillStore

    store = _SpillStore(str(tmp_path / "spill"))
    m0 = ("phasor_mapping", "Phase", 1, 0)
    m1 = ("phasor_mapping", "Phase", 1, 1)
    store.append(m0, np.array([1.0, 2.0, 3.0]))
    store.append(m1, np.array([4.0, 5.0, 6.0]))
    aggregate = {"hist_store": store}
    groups = {1: [m0, m1]}

    items = list(
        BatchAnalysisWidget._hist_group_items(
            aggregate, "phasor_mapping", "Phase", groups, streaming=True
        )
    )
    assert len(items) == 1
    key, file_arrays = items[0]
    assert key == 1
    # Two separate per-file arrays (not one pooled array) -> SD band possible.
    assert len(file_arrays) == 2
    assert sorted(np.concatenate(file_arrays)) == [1, 2, 3, 4, 5, 6]


def test_selection_labels_use_cursor_colors():
    """Selection label images are colored with the cursor colors."""
    layer = _make_phasor_layer(harmonic=[1])
    selection = {
        "harmonic": 1,
        "mode": "manual",
        "cursors": [
            {
                "type": "circular",
                "g": 0.5,
                "s": 0.3,
                "radius": 0.1,
                "color": "#ff0000",
            },
            {
                "type": "circular",
                "g": 0.3,
                "s": 0.3,
                "radius": 0.1,
                "color": "#00ff00",
            },
        ],
    }
    outputs = _apply_selection(layer, selection)
    assert len(outputs) == 1
    color_dict = outputs[0].colormap.color_dict
    assert tuple(round(c, 3) for c in color_dict[1][:3]) == (1.0, 0.0, 0.0)
    assert tuple(round(c, 3) for c in color_dict[2][:3]) == (0.0, 1.0, 0.0)
    assert tuple(color_dict[None]) == (0.0, 0.0, 0.0, 0.0)


def test_linear_second_colormap_is_reversed_first():
    """Linear Projection's 2nd component colormap is the reversed 1st."""
    layer = _make_phasor_layer(harmonic=[1])
    config = {
        "analysis_type": "linear",
        "names": ["A", "B"],
        "colormaps": ["turbo", "viridis"],
        "component_real": [0.8, 0.2],
        "component_imag": [0.3, 0.3],
        "harmonic": 1,
        "contrast_limits": None,
    }
    outputs = _apply_component_fraction(layer, config)
    first = np.asarray(outputs[0].colormap.colors)
    second = np.asarray(outputs[1].colormap.colors)
    np.testing.assert_allclose(second, first[::-1])


def test_export_colorbar_checkbox_default_on(qtbot, make_viewer_model):
    """The analysis-image colorbar checkbox defaults to on."""
    widget = BatchAnalysisWidget(make_viewer_model())
    qtbot.addWidget(widget)
    assert widget.export_colorbar_checkbox.isChecked()


def test_suffix_edit_empty_by_default_with_hint(qtbot, make_viewer_model):
    """The filename suffix field starts empty and shows placeholder hint text."""
    widget = BatchAnalysisWidget(make_viewer_model())
    qtbot.addWidget(widget)
    assert widget.suffix_edit.text() == ""
    assert widget.suffix_edit.placeholderText() != ""


# -- Pure-helper edge cases ------------------------------------------------


def _plain_layer(shape=(4, 4)):
    """An Image layer with no phasor metadata."""
    return Image(np.zeros(shape, dtype=float), name="plain")


def test_scan_folder_missing_folder_returns_empty():
    # Non-recursive scan of a non-existent directory swallows the OSError.
    assert scan_folder("/this/path/does/not/exist", recursive=False) == {}


def test_parse_harmonics_skips_empty_parts():
    assert parse_harmonics("1,,2") == [1, 2]
    assert parse_harmonics(" , ") is None


def test_load_mask_array_npy_and_multichannel(tmp_path):
    # An RGB .npy mask is read and reduced to 2-D (any channel non-zero).
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[1, 1, 0] = 255
    rgb[2, 2, 2] = 10
    path = tmp_path / "mask.npy"
    np.save(path, rgb)
    mask = _load_mask_array(str(path))
    assert mask.shape == (4, 4)
    assert mask[1, 1] and mask[2, 2]
    assert not mask[0, 0]


def test_apply_image_mask_skips_missing_coords():
    # Layer with intensity data but no G/S metadata: masking still applies to
    # the image data and skips the absent coordinate arrays.
    layer = _plain_layer()
    layer.data = np.ones((4, 4))
    mask = np.zeros((4, 4))
    mask[0, 0] = 1
    _apply_image_mask(layer, mask)
    assert np.isnan(layer.data[1, 1])
    assert layer.data[0, 0] == 1
    assert "G" not in layer.metadata


def test_select_harmonic_arrays_without_metadata():
    assert _select_harmonic_arrays(_plain_layer(), 1) == (None, None)


def test_select_harmonic_arrays_falls_back_without_harmonics():
    layer = _plain_layer((2, 3))
    layer.metadata["G"] = np.ones((2, 3))
    layer.metadata["S"] = np.zeros((2, 3))
    real, imag = _select_harmonic_arrays(layer, 1)
    assert np.array_equal(real, np.ones((2, 3)))
    assert np.array_equal(imag, np.zeros((2, 3)))


def test_apply_component_fraction_no_data_returns_empty():
    dummy = np.zeros((1, 2))
    base = {
        "names": ["A", "B"],
        "component_real": dummy,
        "component_imag": dummy,
        "analysis_type": "linear",
        "harmonic": 1,
    }
    # Linear, single harmonic, layer lacks phasor coordinates.
    assert _apply_component_fraction(_plain_layer(), base) == []
    # Multi-harmonic path also bails out when a harmonic has no data.
    multi = dict(base, harmonics=[1, 2])
    assert _apply_component_fraction(_plain_layer(), multi) == []


def test_apply_component_fraction_fit_without_mean():
    layer = _make_phasor_layer()
    layer.metadata.pop("original_mean", None)
    dummy = np.zeros((1, 3))
    comp = {
        "names": ["A", "B", "C"],
        "component_real": dummy,
        "component_imag": dummy,
        "analysis_type": "fit",
        "harmonic": 1,
    }
    assert _apply_component_fraction(layer, comp) == []


def test_reversed_colormap_variants():
    # Empty name returns it unchanged.
    assert _reversed_colormap("") == ""
    assert _reversed_colormap(None) is None
    # A real colormap is reversed.
    reversed_cmap = _reversed_colormap("viridis")
    assert reversed_cmap != "viridis"
    assert hasattr(reversed_cmap, "colors")
    # An invalid name falls back to the original string.
    assert (
        _reversed_colormap("not_a_real_colormap_xyz")
        == "not_a_real_colormap_xyz"
    )


def test_make_output_image_ignores_degenerate_contrast():
    img = _make_output_image(
        np.ones((2, 2)), "out", contrast_limits=(1.0, 1.0)
    )
    assert "contrast_limits" not in (img.metadata or {})


def test_apply_mapping_fret_selection_no_data_return_empty():
    layer = _plain_layer()
    assert _apply_phasor_mapping(layer, {"harmonic": 1}) == []
    assert (
        _apply_fret(
            layer,
            {
                "harmonic": 1,
                "frequency": 80.0,
                "donor_lifetime": 4.0,
                "donor_background": 0.0,
                "background_real": 0.0,
                "background_imag": 0.0,
                "donor_fretting": 1.0,
            },
        )
        == []
    )
    assert _apply_selection(layer, {"harmonic": 1, "cursors": []}) == []


def test_selection_statistics_no_data_returns_empty():
    layer = _plain_layer()
    assert _selection_statistics(layer, {"harmonic": 1}, None) == []


def test_cursor_rgba_color_and_palette_fallback():
    # Explicit color is parsed.
    red = _cursor_rgba("#ff0000", 0)
    assert red[:3] == (1.0, 0.0, 0.0)
    # None falls back to the default palette cycling by index.
    fallback = _cursor_rgba(None, 0)
    assert len(fallback) == 4
    # A custom palette is honored.
    custom = _cursor_rgba(None, 0, palette=["#00ff00"])
    assert custom[:3] == (0.0, 1.0, 0.0)


def test_batch_read_options_rows_and_typed_params(qtbot, monkeypatch):
    widget = BatchReadOptionsWidget()
    qtbot.addWidget(widget)

    # .lif exercises int_or_none (with a value) and str typed params.
    widget.set_extension(".lif")
    image_widget = widget._param_widgets["image"][0]
    image_widget.setText("2")
    dim_widget = widget._param_widgets["dim"][0]
    dim_widget.setText("λ")

    # Add two dynamic kwarg rows, then remove the first.
    widget._add_kwarg_row("alpha", "[1, 2]")
    widget._add_kwarg_row("beta", "not-a-literal")
    assert len(widget._kwargs_rows) == 2
    widget._remove_kwarg_row(widget._kwargs_rows[0])
    assert len(widget._kwargs_rows) == 1

    options = widget.get_reader_options()
    assert options["image"] == 2
    assert options["dim"] == "λ"
    # The literal is eval'd, the non-literal stays a string.
    assert options["beta"] == "not-a-literal"

    # .json exercises the str_or_none branch left at its default (None).
    widget.set_extension(".json")
    assert widget.get_reader_options()["dtype"] is None


def test_copy_settings_dialog_selection(qtbot, monkeypatch):
    dialog = CopySettingsDialog(["Layer A", "Layer B"])
    qtbot.addWidget(dialog)

    # Selecting a layer reports the layer and no file.
    dialog.layer_combo.setCurrentIndex(dialog.layer_combo.findData("Layer B"))
    assert dialog.selected_layer() == "Layer B"
    assert dialog.selected_file() == ""

    # Browsing to a file records it and resets the layer combo.
    monkeypatch.setattr(
        "napari_phasors._batch_analysis.QFileDialog.getOpenFileName",
        lambda *a, **k: ("/tmp/settings.ome.tif", ""),
    )
    dialog._browse()
    assert dialog.selected_file() == "/tmp/settings.ome.tif"
    assert dialog.selected_layer() is None

    # Re-selecting a layer clears the chosen file.
    dialog.layer_combo.setCurrentIndex(dialog.layer_combo.findData("Layer A"))
    assert dialog.selected_file() == ""


# -- Stateless overlay / draw helpers (shared with batch export) -----------


def _agg_axes():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    return plt, fig, ax


def test_draw_components_overlay_two_with_colormap_line():
    from napari_phasors.components_tab import draw_components_overlay

    plt, fig, ax = _agg_axes()
    # Linear projection with a colormap gradient line (<=32 interpolated stops).
    draw_components_overlay(
        ax,
        [0.2, 0.8],
        [0.3, 0.4],
        names=["A", "B"],
        analysis_type="Linear Projection",
        settings={
            "show_colormap_line": True,
            "fractions_colormap": ["#000000", "#ffffff"],
            "line_offset": 0.05,
        },
    )
    # A >32 list takes the ListedColormap branch instead.
    draw_components_overlay(
        ax,
        [0.2, 0.8],
        [0.3, 0.4],
        settings={
            "show_colormap_line": True,
            "fractions_colormap": [f"C{i % 10}" for i in range(40)],
        },
    )
    assert ax.collections
    plt.close(fig)


def test_draw_components_overlay_plain_line_and_polygon():
    from napari_phasors.components_tab import draw_components_overlay

    plt, fig, ax = _agg_axes()
    # Two components, no colormap -> a plain connecting line.
    draw_components_overlay(ax, [0.1, 0.9], [0.2, 0.5])
    # Three components -> a closed polygon patch, with hidden dots/labels.
    draw_components_overlay(
        ax,
        [0.1, 0.9, 0.5],
        [0.2, 0.5, 0.8],
        settings={"show_dots": False, "show_labels": False},
    )
    assert ax.patches
    plt.close(fig)


def test_draw_selection_overlay_cursor_types_and_cluster():
    from napari_phasors.selection_tab import draw_selection_overlay

    plt, fig, ax = _agg_axes()
    # Cluster mode draws no static cursors (early return).
    assert draw_selection_overlay(ax, [], mode="cluster") is None
    cursors = [
        {
            "type": "circular",
            "g": 0.3,
            "s": 0.3,
            "radius": 0.1,
            "color": "#ff0000",
        },
        {
            "type": "elliptic",
            "g": 0.5,
            "s": 0.4,
            "radius": 0.1,
            "radius_minor": 0.05,
            "angle": 30,
            "color": (1.0, 0.0, 0.0, 1.0),
        },
        {
            "type": "polar",
            "phase_min": 10,
            "phase_max": 40,
            "modulation_min": 0.6,
            "modulation_max": 0.6,
            "color": "green",
        },
    ]
    draw_selection_overlay(ax, cursors, mode="cursor")
    assert len(ax.patches) == 3
    plt.close(fig)


def test_draw_fret_trajectory_overlay_modes():
    import numpy as np

    from napari_phasors.fret_tab import draw_fret_trajectory_overlay

    plt, fig, ax = _agg_axes()
    real = np.linspace(0.9, 0.1, 200)
    imag = np.linspace(0.1, 0.4, 200)
    eff = np.linspace(0, 1, 200)
    # Colormap mode, interpolated list (<=32).
    draw_fret_trajectory_overlay(
        ax, real, imag, eff, settings={"fret_colormap": ["#000000", "#ffffff"]}
    )
    # Colormap mode, large list -> ListedColormap.
    draw_fret_trajectory_overlay(
        ax,
        real,
        imag,
        eff,
        settings={"fret_colormap": [f"C{i % 10}" for i in range(40)]},
    )
    # Colormap mode, non-list colormap -> default turbo.
    draw_fret_trajectory_overlay(
        ax, real, imag, eff, settings={"fret_colormap": "turbo"}
    )
    # Plain (no colormap) mode draws a single gray line.
    draw_fret_trajectory_overlay(
        ax, real, imag, eff, settings={"use_colormap": False}
    )
    assert ax.collections or ax.lines
    assert len(ax.patches) >= 2  # donor + background dots each call
    plt.close(fig)


def test_draw_phasor_mesh_full_quadrant_and_modulation():
    from napari_phasors.phasor_mapping_tab import draw_phasor_mesh

    plt, fig, ax = _agg_axes()
    # Non-semicircle geometry (full quadrant) with the Modulation field and a
    # supplied modulation range -> exercises the wrap + range branches.
    image = draw_phasor_mesh(
        ax,
        "Modulation",
        semicircle=False,
        modulation_range=(0.2, 0.9),
        colormap="viridis",
    )
    assert image is not None
    # Unknown colormap name falls back to viridis; no range -> auto vmin/vmax.
    image2 = draw_phasor_mesh(
        ax, "Modulation", semicircle=True, colormap="not_a_real_cmap_xyz"
    )
    assert image2 is not None
    plt.close(fig)


def test_normalize_rgb_and_solid_contour_cmap():

    from napari_phasors._utils import make_solid_contour_cmap, normalize_rgb

    assert normalize_rgb(None) is None
    # Matplotlib color string.
    assert normalize_rgb("red") == pytest.approx((1.0, 0.0, 0.0))
    # 0-255 array is scaled into 0-1.
    assert normalize_rgb([255, 0, 0]) == pytest.approx((1.0, 0.0, 0.0))
    # Already-normalized array is passed through.
    assert normalize_rgb([0.5, 0.25, 0.0]) == pytest.approx((0.5, 0.25, 0.0))
    cmap = make_solid_contour_cmap("solid", "#ff0000")
    # Low end is blended toward white, high end is the target color.
    assert cmap(1.0)[:3] == pytest.approx((1.0, 0.0, 0.0))
    assert cmap(0.0)[0] > cmap(0.0)[1]  # still reddish at the low end


def test_null_progress_supports_progress_api():
    from napari_phasors._utils import _NullProgress

    with _NullProgress() as prog:
        prog.update(1)
        prog.set_description("x")
        prog.increment()
        assert list(prog) == []
        prog.close()


# -- Copy-settings -> UI population (all analysis tabs) --------------------


def test_apply_settings_to_ui_populates_all_tabs(qtbot, make_viewer_model):
    """A full settings dict drives calibration, filter, components, mapping,
    FRET and selection tabs in one pass."""

    widget = BatchAnalysisWidget(make_viewer_model())
    qtbot.addWidget(widget)

    settings = {
        "frequency": [80.0],
        "calibrated": True,
        "calibration_phase": 0.1,
        "calibration_modulation": 0.9,
        # Wavelet branch + upper threshold (median is covered elsewhere).
        "filter": {"method": "wavelet", "sigma": 2.5, "levels": 3},
        "threshold_method": "otsu",
        "threshold": 0.5,
        "threshold_upper": 9.0,
        "component_analysis": {
            "components": {
                "0": {
                    "name": "C1",
                    "gs_harmonics": {
                        "1": {"g": 0.2, "s": 0.3},
                        "2": {"g": 0.15, "s": 0.25},
                    },
                },
                "1": {
                    "name": "C2",
                    "gs_harmonics": {
                        "1": {"g": 0.7, "s": 0.4},
                        "2": {"g": 0.6, "s": 0.35},
                    },
                },
            }
        },
        "lifetime": {"lifetime_type": "Apparent Phase Lifetime"},
        "fret_analysis": {
            "donor_lifetime": 3.5,
            "donor_background": 0.1,
            "donor_fretting": 0.8,
            "background_real": 0.05,
            "background_imag": 0.02,
        },
        "selections": {
            "circular_cursors": [
                {"g": 0.3, "s": 0.3, "radius": 0.1, "color": [255, 0, 0, 255]}
            ],
            "elliptical_cursors": [
                {
                    "g": 0.5,
                    "s": 0.4,
                    "radius": 0.1,
                    "radius_minor": 0.05,
                    "angle": 30.0,
                    "color": "#00ff00",
                }
            ],
            "polar_cursors": [
                {
                    "phase_min": 10.0,
                    "phase_max": 40.0,
                    "modulation_min": 0.5,
                    "modulation_max": 0.8,
                }
            ],
        },
        "semi_circle": True,
        "log_scale": True,
        "bins": 128,
        "batch_group_config": {
            "mode": "Grouped",
            "assignments": {"a.ome.tif": 1},
            "group_names": {"1": "G1"},
            "group_colors": {"1": "#123456"},
        },
    }
    widget._apply_settings_to_ui(settings)

    assert widget.calibration_group.isChecked()
    assert widget._copied_calibration is not None
    assert widget.filter_group.isChecked()
    assert widget.filter_method_combo.currentText() == "Wavelet"
    assert widget.wavelet_sigma_spin.value() == 2.5
    assert widget.wavelet_levels_spin.value() == 3
    assert widget.threshold_max_spin.value() == 9.0
    assert widget.components_group.isChecked()
    assert len(widget._component_rows) == 2
    assert widget.fret_group.isChecked()
    assert widget.fret_donor_lifetime_spin.value() == 3.5
    assert widget.fret_background_spin.value() == 0.1
    assert widget.fret_fretting_spin.value() == 0.8
    assert widget.mapping_group.isChecked()
    assert widget.selection_group.isChecked()
    assert len(widget._cursor_rows) == 3
    assert widget._group_config["mode"] == "Grouped"
    assert widget._group_config["group_names"] == {1: "G1"}
    assert widget.calib_frequency_spin.text() == "80.0"


def test_apply_settings_to_ui_ignores_empty_and_bad_frequency(
    qtbot, make_viewer_model
):
    widget = BatchAnalysisWidget(make_viewer_model())
    qtbot.addWidget(widget)
    # Empty settings is a no-op.
    widget._apply_settings_to_ui({})
    widget._apply_settings_to_ui(None)
    # A non-numeric frequency is swallowed, and a single component is ignored.
    widget._apply_settings_to_ui(
        {
            "frequency": "not-a-number",
            "component_analysis": {
                "components": {
                    "0": {"gs_harmonics": {"1": {"g": 0.1, "s": 0.2}}}
                }
            },
        }
    )
    assert not widget.components_group.isChecked()


def test_component_style_dialogs_apply_on_accept(
    qtbot, make_viewer_model, monkeypatch
):
    from qtpy.QtWidgets import QDialog

    import napari_phasors._batch_analysis as ba

    widget = BatchAnalysisWidget(make_viewer_model())
    qtbot.addWidget(widget)

    monkeypatch.setattr(ba.QDialog, "exec", lambda self: QDialog.Accepted)
    widget._open_component_line_style_dialog()
    widget._open_component_label_style_dialog()
    # Accepting writes the (default) values back into the style dicts.
    assert "line_width" in widget._component_line_style
    assert "fontsize" in widget._component_label_style

    # A rejected dialog leaves the styles untouched.
    before = dict(widget._component_line_style)
    monkeypatch.setattr(ba.QDialog, "exec", lambda self: QDialog.Rejected)
    widget._open_component_line_style_dialog()
    assert widget._component_line_style == before


def test_remove_component_and_cursor_rows(qtbot, make_viewer_model):
    widget = BatchAnalysisWidget(make_viewer_model())
    qtbot.addWidget(widget)

    # Components start with two rows; a third can be added then removed.
    widget._add_component_row("Extra", 0.4, 0.4)
    assert len(widget._component_rows) == 3
    widget._remove_component_row(widget._component_rows[-1])
    assert len(widget._component_rows) == 2
    # The minimum of two rows cannot be removed.
    widget._remove_component_row(widget._component_rows[-1])
    assert len(widget._component_rows) == 2

    # Cursor rows can be added and removed freely.
    widget._add_cursor_row()
    widget._add_cursor_row()
    n = len(widget._cursor_rows)
    widget._remove_cursor_row(widget._cursor_rows[-1])
    assert len(widget._cursor_rows) == n - 1


def test_has_extra_outputs_branches(qtbot, make_viewer_model):
    widget = BatchAnalysisWidget(make_viewer_model())
    qtbot.addWidget(widget)

    def _reset():
        widget.plot_individual_checkbox.setChecked(False)
        widget.plot_combined_checkbox.setChecked(False)
        for g in (
            widget.components_group,
            widget.mapping_group,
            widget.fret_group,
            widget.selection_group,
        ):
            g.setChecked(False)

    _reset()
    assert widget._has_extra_outputs() is False

    # A combined phasor plot counts as an extra output.
    widget.plot_combined_checkbox.setChecked(True)
    assert widget._has_extra_outputs() is True

    # A per-tab plot toggle counts when its group is enabled.
    _reset()
    widget.components_group.setChecked(True)
    widget.components_plot_toggle.setChecked(True)
    assert widget._has_extra_outputs() is True

    # Selection statistics count even without a plot.
    _reset()
    widget.selection_group.setChecked(True)
    widget.selection_plot_toggle.setChecked(False)
    widget.selection_stats_checkbox.setChecked(True)
    assert widget._has_extra_outputs() is True

    # A per-tab stats export counts.
    _reset()
    widget.mapping_group.setChecked(True)
    widget.mapping_plot_toggle.setChecked(False)
    widget.mapping_export_controls["stats"].setChecked(True)
    assert widget._has_extra_outputs() is True


def test_collect_filter_kwargs_wavelet_and_manual(qtbot, make_viewer_model):
    widget = BatchAnalysisWidget(make_viewer_model())
    qtbot.addWidget(widget)

    widget.filter_method_combo.setCurrentText("Wavelet")
    widget.wavelet_sigma_spin.setValue(1.5)
    widget.wavelet_levels_spin.setValue(4)
    widget.threshold_method_combo.setCurrentText("Manual")
    widget.threshold_min_spin.setValue(0.5)
    widget.threshold_max_spin.setValue(5.0)

    kwargs = widget._collect_filter_kwargs()
    assert kwargs["filter_method"] == "wavelet"
    assert kwargs["sigma"] == 1.5
    assert kwargs["levels"] == 4
    assert kwargs["threshold_method"] == "manual"
    assert kwargs["threshold"] == 0.5
    assert kwargs["threshold_upper"] == 5.0


def test_masked_signal_mean_applies_mask():
    """The masked mean averages only the pixels kept by the mask."""
    from napari_phasors._batch_analysis import _masked_signal_mean

    full = np.zeros((3, 2, 2))
    full[:, 0, 0] = [1, 2, 3]
    full[:, 0, 1] = [10, 20, 30]
    full[:, 1, 0] = [100, 200, 300]
    full[:, 1, 1] = [1000, 2000, 3000]

    mask = {"array": np.array([[1, 0], [0, 0]]), "invert": False}
    assert np.allclose(_masked_signal_mean(full, 0, mask), [1, 2, 3])

    # No mask -> mean over every pixel.
    expected_all = np.mean(
        [[1, 10, 100, 1000], [2, 20, 200, 2000], [3, 30, 300, 3000]], axis=1
    )
    assert np.allclose(_masked_signal_mean(full, 0, None), expected_all)

    # Invert keeps the three pixels outside the mask.
    inv = {"array": np.array([[1, 0], [0, 0]]), "invert": True}
    expected_inv = np.mean(
        [[10, 100, 1000], [20, 200, 2000], [30, 300, 3000]], axis=1
    )
    assert np.allclose(_masked_signal_mean(full, 0, inv), expected_inv)

    # A mismatched mask shape falls back to averaging every pixel.
    bad = {"array": np.ones((3, 3)), "invert": False}
    assert np.allclose(_masked_signal_mean(full, 0, bad), expected_all)


def test_signal_export_ometiff_outputs(qtbot, make_viewer_model, tmp_path):
    """OME-TIFFs written by napari-phasors export individual + combined signals."""
    in_root = tmp_path / "in"
    in_root.mkdir()
    out_root = tmp_path / "out"
    out_root.mkdir()
    for name in ("a", "b"):
        write_ome_tiff(
            str(in_root / f"{name}.ome.tif"),
            _make_phasor_layer(name=name, harmonic=[1, 2]),
        )

    widget = BatchAnalysisWidget(make_viewer_model())
    qtbot.addWidget(widget)
    widget._input_folder = str(in_root)
    widget._rescan()
    idx = widget.format_combobox.findData(".ome.tif")
    widget.format_combobox.setCurrentIndex(idx)
    widget._export_folder = str(out_root)
    widget.export_ometiff_checkbox.setChecked(False)

    assert widget._signal_available is True
    widget.signal_group.setChecked(True)
    widget.signal_individual_checkbox.setChecked(True)
    widget.signal_combined_checkbox.setChecked(True)

    widget.run_batch()

    individual = list(
        (out_root / "Individual image analysis" / "Signal Plots").rglob(
            "*_signal.png"
        )
    )
    assert len(individual) == 2
    combined = (
        out_root / "Combined analysis" / "Signal Plots" / "combined_signal.png"
    )
    assert combined.exists()
    assert "2/2 files processed" in widget.status_label.text()


def test_signal_export_unavailable_for_third_party_ometiff(
    qtbot, make_viewer_model, tmp_path
):
    """An OME-TIFF without a stored signal locks the signal tab."""
    in_root = tmp_path / "in"
    in_root.mkdir()
    layer = _make_phasor_layer(name="ext", harmonic=[1, 2])
    del layer.metadata["summed_signal"]  # emulate a file written elsewhere
    write_ome_tiff(str(in_root / "ext.ome.tif"), layer)

    widget = BatchAnalysisWidget(make_viewer_model())
    qtbot.addWidget(widget)
    widget._input_folder = str(in_root)
    widget._rescan()
    idx = widget.format_combobox.findData(".ome.tif")
    widget.format_combobox.setCurrentIndex(idx)

    assert widget._signal_available is False
    assert "unavailable" in widget._signal_status.text().lower()
    assert widget._signal_export_requested() is False


def test_signal_export_raw_tif(qtbot, make_viewer_model, tmp_path):
    """Raw files retain the per-pixel signal and export individual plots."""
    import tifffile

    in_root = tmp_path / "in"
    in_root.mkdir()
    out_root = tmp_path / "out"
    out_root.mkdir()
    raw = make_raw_flim_data(
        n_time_bins=16, shape=(4, 4), time_constants=[0.1, 1, 10]
    ).astype(np.float32)
    tifffile.imwrite(str(in_root / "raw_a.tif"), raw)
    tifffile.imwrite(str(in_root / "raw_b.tif"), raw)

    widget = BatchAnalysisWidget(make_viewer_model())
    qtbot.addWidget(widget)
    widget._input_folder = str(in_root)
    widget._rescan()
    idx = widget.format_combobox.findData(".tif")
    widget.format_combobox.setCurrentIndex(idx)
    widget._export_folder = str(out_root)
    widget.export_ometiff_checkbox.setChecked(False)
    widget.harmonics_edit.setText("1")

    assert widget._signal_available is True
    widget.signal_group.setChecked(True)
    widget.signal_individual_checkbox.setChecked(True)

    widget.run_batch()

    individual = list(
        (out_root / "Individual image analysis" / "Signal Plots").rglob(
            "*_signal.png"
        )
    )
    assert len(individual) == 2


def _fake_channel_results(profiles_by_channel):
    """Build a batch ``results`` list emulating one file's per-channel layers."""
    results = []
    for channel, profile in profiles_by_channel:
        layer = Image(
            np.zeros((2, 2)),
            name=f"f Intensity Image: Channel {channel}",
            metadata={
                "settings": {"channel": channel},
                "_signal_profile": np.asarray(profile, dtype=float),
            },
        )
        results.append((layer, []))
    return results


def _signal_export_widget(make_viewer_model, qtbot, out_root, channel_mode):
    widget = BatchAnalysisWidget(make_viewer_model())
    qtbot.addWidget(widget)
    widget._export_folder = str(out_root)
    widget._signal_export_cfg = {
        "individual": True,
        "combined": True,
        "png": True,
        "csv": False,
        "normalize": "none",
        "channel_mode": channel_mode,
        "color": "#1f77b4",
        "white_background": True,
        "legend": True,
    }
    widget._signal_combined = {}
    return widget


def test_signal_export_channels_separate(qtbot, make_viewer_model, tmp_path):
    out_root = tmp_path / "out"
    out_root.mkdir()
    widget = _signal_export_widget(
        make_viewer_model, qtbot, out_root, "separate"
    )
    for name in ("a", "b"):
        widget._emit_signal_outputs(
            f"{name}.tif",
            _fake_channel_results([(0, [1, 2, 3, 4]), (1, [4, 3, 2, 1])]),
            ".tif",
            "",
            False,
        )
    widget._write_signal_combined()

    ind = out_root / "Individual image analysis" / "Signal Plots"
    assert (ind / "a_Channel_0_signal.png").exists()
    assert (ind / "a_Channel_1_signal.png").exists()
    assert not (ind / "a_signal.png").exists()
    comb = out_root / "Combined analysis" / "Signal Plots"
    assert (comb / "combined_signal_Channel_0.png").exists()
    assert (comb / "combined_signal_Channel_1.png").exists()
    assert not (comb / "combined_signal.png").exists()


def test_signal_export_channels_together(qtbot, make_viewer_model, tmp_path):
    out_root = tmp_path / "out"
    out_root.mkdir()
    widget = _signal_export_widget(
        make_viewer_model, qtbot, out_root, "together"
    )
    for name in ("a", "b"):
        widget._emit_signal_outputs(
            f"{name}.tif",
            _fake_channel_results([(0, [1, 2, 3, 4]), (1, [4, 3, 2, 1])]),
            ".tif",
            "",
            False,
        )
    widget._write_signal_combined()

    ind = out_root / "Individual image analysis" / "Signal Plots"
    assert (ind / "a_signal.png").exists()  # channels overlaid in one figure
    assert not (ind / "a_Channel_0_signal.png").exists()
    comb = out_root / "Combined analysis" / "Signal Plots"
    assert (comb / "combined_signal.png").exists()  # one overlaid figure
    assert not (comb / "combined_signal_Channel_0.png").exists()


def test_signal_export_csv_only(qtbot, make_viewer_model, tmp_path):
    """CSV-only export writes signal data as CSV and no PNG."""
    out_root = tmp_path / "out"
    out_root.mkdir()
    widget = _signal_export_widget(
        make_viewer_model, qtbot, out_root, "together"
    )
    widget._signal_export_cfg["png"] = False
    widget._signal_export_cfg["csv"] = True
    for name in ("a", "b"):
        widget._emit_signal_outputs(
            f"{name}.tif",
            _fake_channel_results([(0, [1, 2, 3, 4])]),
            ".tif",
            "",
            False,
        )
    widget._write_signal_combined()

    ind = out_root / "Individual image analysis" / "Signal Plots"
    assert (ind / "a_signal.csv").exists()
    assert not (ind / "a_signal.png").exists()
    comb = out_root / "Combined analysis" / "Signal Plots"
    combined_csv = comb / "combined_signal.csv"
    assert combined_csv.exists()
    assert not (comb / "combined_signal.png").exists()
    # Header carries the bin column plus a mean/std pair for the single group.
    header = combined_csv.read_text().splitlines()[0].split(",")
    assert header[0] == "bin"
    assert any("mean" in h for h in header)
    assert any("std" in h for h in header)
