import json
import sys

import numpy as np
from phasorpy.io import (
    phasor_from_ometiff,
    signal_from_fbd,
    signal_from_lsm,
    signal_from_ptu,
    signal_from_sdt,
)

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
    assert (
        layer_data_tuple[1]["name"] == "test_file Intensity Image: Channel 0"
    )
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
    if "description" in attrs.keys():
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


# TODO: Add tests for .tif files
