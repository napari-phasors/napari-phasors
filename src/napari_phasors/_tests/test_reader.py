import json

import numpy as np
import pytest
import xarray as xr
from phasorpy.datasets import fetch
from phasorpy.io import (
    phasor_from_ometiff,
    signal_from_fbd,
    signal_from_lsm,
    signal_from_ptu,
    signal_from_sdt,
)

import napari_phasors._reader as reader_module
from napari_phasors import napari_get_reader
from napari_phasors._tests.test_data_utils import get_test_file_path


def test_reader_ptu():
    """Test reading a PTU file"""
    ptu_file = get_test_file_path("test_file.ptu")
    reader = napari_get_reader(ptu_file)
    assert callable(reader)
    layer_data_list = reader(ptu_file)
    assert isinstance(layer_data_list, list) and len(layer_data_list) == 1
    # First Channel
    layer_data_tuple = layer_data_list[0]
    assert isinstance(layer_data_tuple, tuple) and len(layer_data_tuple) == 2
    assert isinstance(layer_data_tuple[0], np.ndarray) and isinstance(
        layer_data_tuple[1], dict
    )
    assert layer_data_tuple[0].shape == (256, 256)
    assert "name" in layer_data_tuple[1] and "metadata" in layer_data_tuple[1]
    assert layer_data_tuple[1]["name"] == "test_file Intensity Image"
    metadata = layer_data_tuple[1]["metadata"]
    assert "G" in metadata
    assert "S" in metadata
    assert "G_original" in metadata
    assert "S_original" in metadata
    assert "harmonics" in metadata
    assert "original_mean" in metadata
    assert "settings" in metadata

    signal_data = signal_from_ptu(ptu_file, frame=-1)
    signal_data = np.sum(signal_data, axis=(0, 1))
    signal_from_metadata = np.array(
        layer_data_tuple[1]["metadata"]["summed_signal"]
    )
    np.testing.assert_array_almost_equal(signal_data, signal_from_metadata)

    assert layer_data_tuple[1]["metadata"]["settings"]["channel"] == 0

    # Check phasor arrays
    G = metadata["G"]
    S = metadata["S"]
    G_original = metadata["G_original"]
    S_original = metadata["S_original"]
    harmonics = metadata["harmonics"]

    # Check shapes - G and S should have shape (n_harmonics, height, width)
    assert G.shape == (2, 256, 256)
    assert S.shape == (2, 256, 256)
    assert G_original.shape == (2, 256, 256)
    assert S_original.shape == (2, 256, 256)
    assert list(harmonics) == [1, 2]


def test_reader_ptu_nonzero_channel_label(monkeypatch):
    """PTU channel selection should not assume labels start at zero."""
    data = xr.DataArray(
        np.ones((2, 2, 1, 4), dtype=np.uint16),
        dims=("Y", "X", "C", "H"),
        coords={"Y": [0, 1], "X": [0, 1], "C": [3], "H": [0, 1, 2, 3]},
    )

    monkeypatch.setitem(
        reader_module.extension_mapping["raw"],
        ".ptu",
        lambda path, reader_options: data,
    )

    layer_data_list = reader_module.raw_file_reader("example.ptu")

    assert len(layer_data_list) == 1
    assert layer_data_list[0][1]["name"].endswith("Channel 3")
    assert layer_data_list[0][1]["metadata"]["settings"]["channel"] == 3


@pytest.mark.parametrize("extension", [".ptu", ".fbd", ".json"])
def test_raw_reader_nonzero_channel_label(monkeypatch, extension):
    """Raw readers should use channel position, not channel label, for selection."""
    data = xr.DataArray(
        np.ones((2, 2, 1, 4), dtype=np.uint16),
        dims=("Y", "X", "C", "H"),
        coords={"Y": [0, 1], "X": [0, 1], "C": [3], "H": [0, 1, 2, 3]},
    )

    monkeypatch.setitem(
        reader_module.extension_mapping["raw"],
        extension,
        lambda path, reader_options: data,
    )

    layer_data_list = reader_module.raw_file_reader(f"example{extension}")

    assert len(layer_data_list) == 1
    assert layer_data_list[0][1]["name"].endswith("Channel 3")
    assert layer_data_list[0][1]["metadata"]["settings"]["channel"] == 3


def test_reader_fbd():
    """Test reading a FBD file"""
    fbd_file = get_test_file_path("test_file$EI0S.fbd")
    reader = napari_get_reader(fbd_file)
    assert callable(reader)
    layer_data_list = reader(fbd_file)
    assert isinstance(layer_data_list, list) and len(layer_data_list) == 2
    # First Channel
    layer_data_tuple = layer_data_list[0]
    assert isinstance(layer_data_tuple, tuple) and len(layer_data_tuple) == 2
    assert isinstance(layer_data_tuple[0], np.ndarray) and isinstance(
        layer_data_tuple[1], dict
    )
    assert layer_data_tuple[0].shape == (256, 256)
    assert "name" in layer_data_tuple[1] and "metadata" in layer_data_tuple[1]
    assert (
        layer_data_tuple[1]["name"]
        == "test_file$EI0S Intensity Image: Channel 0"
    )
    metadata = layer_data_tuple[1]["metadata"]
    assert "G" in metadata
    assert "S" in metadata
    assert "G_original" in metadata
    assert "S_original" in metadata
    assert "harmonics" in metadata
    assert "original_mean" in metadata
    assert "settings" in metadata

    signal_channel_0 = signal_from_fbd(fbd_file, frame=-1, channel=0)
    signal_channel_0 = np.sum(signal_channel_0, axis=(0, 1))

    signal_from_metadata = np.array(
        layer_data_tuple[1]["metadata"]["summed_signal"]
    )
    np.testing.assert_array_almost_equal(
        signal_channel_0, signal_from_metadata
    )

    assert layer_data_tuple[1]["metadata"]["settings"]["channel"] == 0

    # Check phasor arrays for channel 0
    G = metadata["G"]
    S = metadata["S"]
    G_original = metadata["G_original"]
    S_original = metadata["S_original"]
    harmonics = metadata["harmonics"]

    assert G.shape == (2, 256, 256)
    assert S.shape == (2, 256, 256)
    assert G_original.shape == (2, 256, 256)
    assert S_original.shape == (2, 256, 256)
    assert list(harmonics) == [1, 2]

    # Second Channel
    layer_data_tuple = layer_data_list[1]
    assert isinstance(layer_data_tuple, tuple) and len(layer_data_tuple) == 2
    assert isinstance(layer_data_tuple[0], np.ndarray) and isinstance(
        layer_data_tuple[1], dict
    )
    assert layer_data_tuple[0].shape == (256, 256)
    assert "name" in layer_data_tuple[1] and "metadata" in layer_data_tuple[1]
    assert (
        layer_data_tuple[1]["name"]
        == "test_file$EI0S Intensity Image: Channel 1"
    )
    metadata = layer_data_tuple[1]["metadata"]
    assert "G" in metadata
    assert "S" in metadata
    assert "G_original" in metadata
    assert "S_original" in metadata
    assert "harmonics" in metadata
    assert "original_mean" in metadata
    assert "settings" in metadata

    signal_channel_1 = signal_from_fbd(fbd_file, frame=-1, channel=1)
    signal_channel_1 = np.sum(signal_channel_1, axis=(0, 1))
    signal_from_metadata = np.array(
        layer_data_tuple[1]["metadata"]["summed_signal"]
    )
    np.testing.assert_array_almost_equal(
        signal_channel_1, signal_from_metadata
    )

    assert layer_data_tuple[1]["metadata"]["settings"]["channel"] == 1

    # Check phasor arrays for channel 1
    G = metadata["G"]
    S = metadata["S"]
    G_original = metadata["G_original"]
    S_original = metadata["S_original"]
    harmonics = metadata["harmonics"]

    assert G.shape == (2, 256, 256)
    assert S.shape == (2, 256, 256)
    assert G_original.shape == (2, 256, 256)
    assert S_original.shape == (2, 256, 256)
    assert list(harmonics) == [1, 2]


def test_reader_sdt():
    """Test reading a sdt file"""
    sdt_file = get_test_file_path('seminal_receptacle_FLIM_single_image.sdt')
    reader = napari_get_reader(sdt_file)
    assert callable(reader)
    layer_data_list = reader(sdt_file)
    assert isinstance(layer_data_list, list) and len(layer_data_list) == 1
    # First Channel
    layer_data_tuple = layer_data_list[0]
    assert isinstance(layer_data_tuple, tuple) and len(layer_data_tuple) == 2
    assert isinstance(layer_data_tuple[0], np.ndarray) and isinstance(
        layer_data_tuple[1], dict
    )
    assert layer_data_tuple[0].shape == (512, 512)
    assert "name" in layer_data_tuple[1] and "metadata" in layer_data_tuple[1]
    assert (
        layer_data_tuple[1]["name"]
        == "seminal_receptacle_FLIM_single_image Intensity Image: Channel 0"
    )
    metadata = layer_data_tuple[1]["metadata"]
    assert "G" in metadata
    assert "S" in metadata
    assert "G_original" in metadata
    assert "S_original" in metadata
    assert "harmonics" in metadata
    assert "original_mean" in metadata
    assert "settings" in metadata

    signal_data = signal_from_sdt(sdt_file)
    signal_data = np.sum(signal_data, axis=(0, 1))

    signal_from_metadata = np.array(
        layer_data_tuple[1]["metadata"]["summed_signal"]
    )
    np.testing.assert_array_almost_equal(signal_data, signal_from_metadata)

    # Check phasor arrays
    G = metadata["G"]
    S = metadata["S"]
    G_original = metadata["G_original"]
    S_original = metadata["S_original"]
    harmonics = metadata["harmonics"]

    assert G.shape == (2, 512, 512)
    assert S.shape == (2, 512, 512)
    assert G_original.shape == (2, 512, 512)
    assert S_original.shape == (2, 512, 512)
    assert list(harmonics) == [1, 2]


def test_reader_lsm():
    """Test reading a LSM file"""
    lsm_file = get_test_file_path("test_file.lsm")
    reader = napari_get_reader(lsm_file)
    assert callable(reader)
    layer_data_list = reader(lsm_file)
    assert isinstance(layer_data_list, list) and len(layer_data_list) == 1
    layer_data_tuple = layer_data_list[0]
    assert isinstance(layer_data_tuple, tuple) and len(layer_data_tuple) == 2
    assert isinstance(layer_data_tuple[0], np.ndarray) and isinstance(
        layer_data_tuple[1], dict
    )
    assert layer_data_tuple[0].shape == (512, 512)
    assert "name" in layer_data_tuple[1] and "metadata" in layer_data_tuple[1]
    assert layer_data_tuple[1]["name"] == "test_file Intensity Image"
    metadata = layer_data_tuple[1]["metadata"]
    assert "G" in metadata
    assert "S" in metadata
    assert "G_original" in metadata
    assert "S_original" in metadata
    assert "harmonics" in metadata
    assert "original_mean" in metadata
    assert "settings" in metadata

    signal_data = signal_from_lsm(lsm_file)
    signal_data = np.sum(signal_data, axis=(1, 2))
    signal_from_metadata = np.array(
        layer_data_tuple[1]["metadata"]["summed_signal"]
    )
    np.testing.assert_array_almost_equal(signal_data, signal_from_metadata)

    # Check phasor arrays
    G = metadata["G"]
    S = metadata["S"]
    G_original = metadata["G_original"]
    S_original = metadata["S_original"]
    harmonics = metadata["harmonics"]

    assert G.shape == (2, 512, 512)
    assert S.shape == (2, 512, 512)
    assert G_original.shape == (2, 512, 512)
    assert S_original.shape == (2, 512, 512)
    assert list(harmonics) == [1, 2]


def test_reader_ometif():
    """Test reading a ome.tif file"""
    ometif_file = get_test_file_path("test_file.ome.tif")
    reader = napari_get_reader(ometif_file)
    assert callable(reader)
    layer_data_list = reader(ometif_file)
    assert isinstance(layer_data_list, list) and len(layer_data_list) == 1
    layer_data_tuple = layer_data_list[0]
    assert isinstance(layer_data_tuple, tuple) and len(layer_data_tuple) == 2
    assert isinstance(layer_data_tuple[0], np.ndarray) and isinstance(
        layer_data_tuple[1], dict
    )
    assert layer_data_tuple[0].shape == (512, 512)
    assert "name" in layer_data_tuple[1] and "metadata" in layer_data_tuple[1]
    assert layer_data_tuple[1]["name"] == "test_file Intensity Image"
    metadata = layer_data_tuple[1]["metadata"]
    assert "G" in metadata
    assert "S" in metadata
    assert "G_original" in metadata
    assert "S_original" in metadata
    assert "harmonics" in metadata
    assert "original_mean" in metadata
    assert "settings" in metadata

    # Check phasor arrays
    G = metadata["G"]
    S = metadata["S"]
    G_original = metadata["G_original"]
    S_original = metadata["S_original"]
    harmonics = metadata["harmonics"]

    assert G.shape == (2, 512, 512)
    assert S.shape == (2, 512, 512)
    assert G_original.shape == (2, 512, 512)
    assert S_original.shape == (2, 512, 512)
    assert list(harmonics) == [1, 2]


def test_reader_ometiff_extension():
    """Test .ome.tiff extension maps to processed reader."""
    reader = napari_get_reader("test_file.ome.tiff")
    assert callable(reader)


def test_reader_tiff_extension():
    """Test .tiff extension maps to raw reader."""
    reader = napari_get_reader("test_file.tiff")
    assert callable(reader)


def test_raw_reader_tiff_does_not_forward_widget_axis_option_to_imread(
    monkeypatch,
):
    """TIFF raw loading should not forward widget-only options (e.g. `phasor_axis`) into the IO/read layer."""

    def fake_imread(path):
        return np.arange(24, dtype=np.float32).reshape(2, 3, 4)

    def fake_phasor_from_signal(signal, axis, harmonic):
        mean_image = np.zeros((2, 4), dtype=np.float32)
        g_image = np.zeros((2, 2, 4), dtype=np.float32)
        s_image = np.zeros((2, 2, 4), dtype=np.float32)
        return mean_image, g_image, s_image

    monkeypatch.setattr(reader_module.tifffile, "imread", fake_imread)
    monkeypatch.setattr(
        reader_module,
        "phasor_from_signal",
        fake_phasor_from_signal,
    )

    layers = reader_module.raw_file_reader(
        "example.tif",
        reader_options={"phasor_axis": 1},
    )

    assert len(layers) == 1
    assert layers[0][0].shape == (2, 4)
    assert layers[0][1]["metadata"]["harmonics"] == [1]


def test_reader_multi_file_raw_dispatches_to_stack_reader(monkeypatch):
    """Test that a list of raw files is handled via raw_file_stack_reader."""
    paths = ["slice_01.lsm", "slice_02.lsm", "slice_03.lsm"]
    expected_layers = [(np.zeros((2, 2, 2)), {"name": "stack"})]

    def fake_stack_reader(
        in_paths,
        reader_options=None,
        harmonics=None,
    ):
        assert in_paths == paths
        assert reader_options == {"foo": "bar"}
        assert harmonics == [1, 2]
        return expected_layers

    monkeypatch.setattr(
        reader_module, "raw_file_stack_reader", fake_stack_reader
    )

    reader = napari_get_reader(
        paths,
        reader_options={"foo": "bar"},
        harmonics=[1, 2],
    )
    assert callable(reader)
    assert reader(paths) == expected_layers


def test_reader_ometif_metadata():
    """Test reading OME-TIFF file and verify metadata settings"""
    ometif_file = get_test_file_path("test_file.ome.tif")
    reader = napari_get_reader(ometif_file)
    assert callable(reader)
    layer_data_list = reader(ometif_file)
    assert isinstance(layer_data_list, list) and len(layer_data_list) == 1

    layer_data_tuple = layer_data_list[0]
    settings = layer_data_tuple[1]["metadata"]["settings"]

    # Test that no channel information is present (OME-TIFF doesn't have channel info like FLIM files)
    assert "channel" not in settings
    assert "frequency" not in settings

    # Test that summed_signal is present and is a list
    _, _, _, attrs = phasor_from_ometiff(ometif_file, harmonic='all')
    signal_data = None
    if "description" in attrs:
        description = json.loads(attrs["description"])
        if len(json.dumps(description)) > 512 * 512:  # Threshold: 256 KB
            raise ValueError("Description dictionary is too large.")
        if "napari_phasors_settings" in description:
            settings = json.loads(description["napari_phasors_settings"])

            # Check if we have summed_signal data
            if 'summed_signal' in settings:
                signal_data = (np.array(settings['summed_signal']),)

    assert "summed_signal" in settings
    assert isinstance(settings["summed_signal"], list)
    np.testing.assert_array_almost_equal(
        signal_data[0], np.array(settings["summed_signal"])
    )

    # Test filter settings
    assert "filter" in settings
    filter_settings = settings["filter"]
    assert filter_settings["method"] == "median"
    assert filter_settings["size"] == 3
    assert filter_settings["repeat"] == 3

    # Test threshold settings
    assert "threshold" in settings
    assert settings["threshold"] == 1.0

    # Test threshold_upper if present (may not be in older files)
    if "threshold_upper" in settings:
        assert isinstance(settings["threshold_upper"], (int, float))
        assert settings["threshold_upper"] >= settings["threshold"]

    # Test circular cursors in metadata (may not be present in older files)
    # If present, verify structure
    if (
        "selections" in settings
        and "circular_cursors" in settings["selections"]
    ):
        circular_cursors = settings["selections"]["circular_cursors"]
        assert isinstance(circular_cursors, list)
        # If there are cursors, verify each has the required fields
        for cursor in circular_cursors:
            assert "g" in cursor
            assert "s" in cursor
            assert "radius" in cursor
            assert "color" in cursor
            assert isinstance(cursor["color"], (list, tuple))
            assert len(cursor["color"]) == 4  # RGBA

    # Test polar cursors in metadata
    if "selections" in settings and "polar_cursors" in settings["selections"]:
        polar_cursors = settings["selections"]["polar_cursors"]
        assert isinstance(polar_cursors, list)
        for cursor in polar_cursors:
            assert "phase_min" in cursor
            assert "phase_max" in cursor
            assert "modulation_min" in cursor
            assert "modulation_max" in cursor
            assert "color" in cursor
            assert len(cursor["color"]) == 4

    # Test elliptical cursors in metadata
    if (
        "selections" in settings
        and "elliptical_cursors" in settings["selections"]
    ):
        elliptical_cursors = settings["selections"]["elliptical_cursors"]
        assert isinstance(elliptical_cursors, list)
        for cursor in elliptical_cursors:
            assert "g" in cursor
            assert "s" in cursor
            assert "radius" in cursor
            assert "radius_minor" in cursor
            assert "angle" in cursor
            assert "color" in cursor
            assert len(cursor["color"]) == 4


def test_reader_czi():
    """Test reading a CZI file"""
    czi_file = get_test_file_path("test_file.czi")
    reader = napari_get_reader(czi_file)
    assert callable(reader)

    layer_data_list = reader(czi_file)

    assert isinstance(layer_data_list, list) and len(layer_data_list) == 1
    layer_data_tuple = layer_data_list[0]

    assert isinstance(layer_data_tuple, tuple) and len(layer_data_tuple) == 2
    assert isinstance(layer_data_tuple[0], np.ndarray) and isinstance(
        layer_data_tuple[1], dict
    )
    assert "name" in layer_data_tuple[1] and "metadata" in layer_data_tuple[1]
    assert layer_data_tuple[1]["name"] == "test_file Intensity Image"

    metadata = layer_data_tuple[1]["metadata"]
    assert "G" in metadata
    assert "S" in metadata

    # Original shape (1, 1, 28, 1, 1, 512, 512, 1) squeezed to (28, 512, 512)
    # Spectral dimension C=28 (index 0)
    # Phasor output for harmonics 1, 2: (2, 512, 512)
    assert metadata["G"].shape == (2, 512, 512)
    assert metadata["S"].shape == (2, 512, 512)

    # Check original mean shape is matched too (512, 512)
    assert metadata["original_mean"].shape == (512, 512)


def test_reader_flif():
    """Test reading a flif file."""
    flif_file = fetch("flimfast.flif")
    reader = napari_get_reader(flif_file)
    assert callable(reader)
    layer_data_list = reader(flif_file)
    assert isinstance(layer_data_list, list) and len(layer_data_list) > 0
    layer_data = layer_data_list[0]
    assert "metadata" in layer_data[1]
    assert "G" in layer_data[1]["metadata"]


def test_reader_bh():
    """Test reading a bh file."""
    bh_file = fetch("simfcs.b&h")
    reader = napari_get_reader(bh_file)
    assert callable(reader)
    layer_data_list = reader(bh_file)
    assert isinstance(layer_data_list, list) and len(layer_data_list) > 0
    layer_data = layer_data_list[0]
    assert "G" in layer_data[1]["metadata"]


def test_reader_bhz():
    """Test reading a bhz file."""
    bhz_file = fetch("simfcs.bhz")
    reader = napari_get_reader(bhz_file)
    assert callable(reader)
    layer_data_list = reader(bhz_file)
    assert isinstance(layer_data_list, list) and len(layer_data_list) > 0
    layer_data = layer_data_list[0]
    assert "G" in layer_data[1]["metadata"]


def test_reader_r64():
    """Test reading a r64 file."""
    r64_file = fetch("simfcs.r64")
    for filename in (r64_file, r64_file.replace(".r64", ".R64")):
        reader = napari_get_reader(filename)
        assert callable(reader)
        # Note: we call the reader with the actual existing file path (r64_file)
        layer_data_list = reader(r64_file)
        assert isinstance(layer_data_list, list) and len(layer_data_list) > 0
        layer_data = layer_data_list[0]
        assert "G" in layer_data[1]["metadata"]


def test_reader_json_imaging():
    """Test reading a JSON imaging file."""
    json_file = fetch("Fluorescein_Calibration_m2_1740751189_imaging.json")
    reader = napari_get_reader(json_file)
    assert callable(reader)
    layer_data_list = reader(json_file)
    assert isinstance(layer_data_list, list) and len(layer_data_list) > 0
    layer_data = layer_data_list[0]
    assert "G" in layer_data[1]["metadata"]


def test_reader_json_phasor():
    """Test reading a JSON phasor file."""
    json_file = fetch("Convallaria_m2_1740751781_phasor_ch1.json")
    reader = napari_get_reader(json_file)
    assert callable(reader)
    layer_data_list = reader(json_file)
    assert isinstance(layer_data_list, list) and len(layer_data_list) > 0
    layer_data = layer_data_list[0]
    assert "G" in layer_data[1]["metadata"]


def test_ambiguous_file_reader_falls_back_to_processed(monkeypatch):
    """Ambiguous reader should fallback to processed when raw fails."""
    expected_layers = [(np.zeros((2, 2)), {"name": "processed"})]

    def fake_raw(*args, **kwargs):
        raise ValueError("raw failed")

    def fake_processed(*args, **kwargs):
        return expected_layers

    monkeypatch.setattr(reader_module, "raw_file_reader", fake_raw)
    monkeypatch.setattr(reader_module, "processed_file_reader", fake_processed)

    result = reader_module.ambiguous_file_reader("ambiguous.json")
    assert result == expected_layers


def test_ambiguous_file_reader_raises_combined_error(monkeypatch):
    """Ambiguous reader should raise an error including both failure contexts."""

    def fake_raw(*args, **kwargs):
        raise ValueError("raw exploded")

    def fake_processed(*args, **kwargs):
        raise TypeError("processed exploded")

    monkeypatch.setattr(reader_module, "raw_file_reader", fake_raw)
    monkeypatch.setattr(reader_module, "processed_file_reader", fake_processed)

    with pytest.raises(
        RuntimeError, match="raw_file_reader error"
    ) as exc_info:
        reader_module.ambiguous_file_reader("ambiguous.json")

    message = str(exc_info.value)
    assert "processed_file_reader error" in message
    assert "raw exploded" in message
    assert "processed exploded" in message


def test_clamp_harmonics():
    from napari_phasors._reader import _clamp_harmonics

    # None defaults to [1, 2]
    assert _clamp_harmonics(None, 4) == [1, 2]
    # 'all' returns all up to n_samples // 2
    assert _clamp_harmonics("all", 6) == [1, 2, 3]
    # Integer clamps to max_h
    assert _clamp_harmonics(3, 4) == [2]
    # List of valid harmonics
    assert _clamp_harmonics([1, 2], 6) == [1, 2]
    # Clamping down excessive harmonics
    assert _clamp_harmonics([1, 4], 4) == [1, 2]
    # Handling duplicates and sorting
    assert _clamp_harmonics([2, 1, 2], 6) == [2, 1]

    # ValueError raised when no valid harmonics remain
    with pytest.raises(ValueError, match="No valid harmonics remain"):
        _clamp_harmonics([], 4)

    with pytest.raises(ValueError, match="No valid harmonics remain"):
        _clamp_harmonics([-1, -2], 4)

    with pytest.raises(ValueError, match="No valid harmonics remain"):
        _clamp_harmonics(["invalid"], 4)

    # ValueError raised when not enough samples
    with pytest.raises(ValueError, match="Not enough samples"):
        _clamp_harmonics([1], 1)


def test_parse_and_call_io_function():
    from napari_phasors._reader import _parse_and_call_io_function

    def func_with_kwargs(path, a=1, **kwargs):
        return {"path": path, "a": a, "kwargs": kwargs}

    def func_without_kwargs(path, a=1):
        return {"path": path, "a": a}

    # Test function that accepts kwargs - custom kwargs should be passed through
    res = _parse_and_call_io_function(
        "dummy_path",
        func_with_kwargs,
        args_defaults={"a": (1, False)},
        reader_options={"a": 42, "custom_kwarg": "hello", "another": [1, 2]},
    )
    assert res == {
        "path": "dummy_path",
        "a": 42,
        "kwargs": {"custom_kwarg": "hello", "another": [1, 2]},
    }

    # Test function that does NOT accept kwargs - custom kwargs should raise ValueError
    with pytest.raises(ValueError, match="Invalid argument 'custom_kwarg'"):
        _parse_and_call_io_function(
            "dummy_path",
            func_without_kwargs,
            args_defaults={"a": (1, False)},
            reader_options={"a": 42, "custom_kwarg": "hello"},
        )


# TODO: Add tests for .tif files
