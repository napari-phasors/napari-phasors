"""Tests for the Batch Analysis widget."""

import os

import numpy as np

from napari_phasors._batch_analysis import (
    BatchAnalysisWidget,
    BatchPipeline,
    apply_pipeline,
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
from napari_phasors._utils import apply_filter_and_threshold
from napari_phasors._writer import write_ome_tiff


def _make_phasor_layer(name="FLIM data", harmonic=None):
    raw = make_raw_flim_data(time_constants=[0.1, 1, 10])
    return make_intensity_layer_with_phasors(raw, harmonic=harmonic, name=name)


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


def test_copy_settings_pipeline_from_layer(qtbot, make_viewer_model):
    viewer = make_viewer_model()
    source = _make_phasor_layer(name="ref")
    source.metadata["settings"] = {
        "filter": {"method": "median", "size": 3, "repeat": 1},
        "threshold": 1.0,
        "threshold_method": "manual",
    }
    viewer.add_layer(source)

    widget = BatchAnalysisWidget(viewer)
    qtbot.addWidget(widget)
    widget.copy_mode_radio.setChecked(True)
    widget.copy_reference_combo.setCurrentText(source.name)

    pipeline = widget.build_pipeline([1, 2])
    assert pipeline.filter is not None
    assert pipeline.filter["filter_method"] == "median"

    target = _make_phasor_layer()
    reference = _make_phasor_layer()
    apply_pipeline(target, pipeline)
    apply_filter_and_threshold(
        reference,
        filter_method="median",
        size=3,
        repeat=1,
        threshold=1.0,
        threshold_method="manual",
    )
    np.testing.assert_array_equal(
        target.metadata["G"], reference.metadata["G"]
    )


def test_manual_calibration_pipeline(qtbot, make_viewer_model):
    viewer = make_viewer_model()
    reference = _make_phasor_layer(name="calibration")
    viewer.add_layer(reference)

    widget = BatchAnalysisWidget(viewer)
    qtbot.addWidget(widget)
    widget.manual_mode_radio.setChecked(True)
    widget.calibration_enable.setChecked(True)
    widget.calib_reference_combo.setCurrentText(reference.name)
    widget.calib_frequency_spin.setValue(80.0)
    widget.calib_lifetime_spin.setValue(2.0)

    pipeline = widget.build_pipeline([1, 2])
    assert pipeline.calibration is not None

    target = _make_phasor_layer()
    before = target.metadata["G"].copy()
    apply_pipeline(target, pipeline)
    assert target.metadata["settings"]["calibrated"] is True
    assert not np.array_equal(before, target.metadata["G"])


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

    widget.filter_enable.setChecked(True)
    widget.filter_method_combo.setCurrentText("Median")
    widget.threshold_method_combo.setCurrentText("Manual")

    widget.run_batch()

    # Outputs preserve the subfolder tree and original names + suffix.
    out_a = out_root / "a_analyzed.ome.tif"
    out_b = out_root / "cond1" / "b_analyzed.ome.tif"
    assert out_a.exists()
    assert out_b.exists()
    assert (out_root / "a_analyzed.csv").exists()

    # Re-readable via the normal reader.
    reader = napari_get_reader(str(out_a))
    assert reader is not None
    result = reader(str(out_a))
    assert result and "G" in result[0][1]["metadata"]
