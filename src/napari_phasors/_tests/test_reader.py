import numpy as np
import pandas as pd
from napari.layers import Labels

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
    assert layer_data_tuple[0].shape == (1, 256, 256)
    assert "name" in layer_data_tuple[1] and "metadata" in layer_data_tuple[1]
    assert (
        layer_data_tuple[1]["name"] == "test_file Intensity Image: Channel 0"
    )
    assert (
        len(layer_data_tuple[1]["metadata"]) == 3
        and "phasor_features_labels_layer" in layer_data_tuple[1]["metadata"]
        and "original_mean" in layer_data_tuple[1]["metadata"]
        and "settings" in layer_data_tuple[1]["metadata"]
    )
    phasor_features = layer_data_tuple[1]["metadata"][
        "phasor_features_labels_layer"
    ]
    assert isinstance(phasor_features, Labels)
    assert phasor_features.data.shape == (1, 256, 256)
    assert isinstance(phasor_features.features, pd.DataFrame)
    assert phasor_features.features.shape == (131072, 6)
    expected_columns = [
        "label",
        "G_original",
        "S_original",
        "G",
        "S",
        "harmonic",
    ]
    actual_columns = phasor_features.features.columns.tolist()
    assert actual_columns == expected_columns
    assert np.unique(phasor_features.features["harmonic"]).tolist() == [1, 2]


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
    assert layer_data_tuple[0].shape == (1, 256, 256)
    assert "name" in layer_data_tuple[1] and "metadata" in layer_data_tuple[1]
    assert (
        layer_data_tuple[1]["name"]
        == "test_file$EI0S Intensity Image: Channel 0"
    )
    assert (
        len(layer_data_tuple[1]["metadata"]) == 3
        and "phasor_features_labels_layer" in layer_data_tuple[1]["metadata"]
        and "original_mean" in layer_data_tuple[1]["metadata"]
        and "settings" in layer_data_tuple[1]["metadata"]
    )
    phasor_features = layer_data_tuple[1]["metadata"][
        "phasor_features_labels_layer"
    ]
    assert isinstance(phasor_features, Labels)
    assert phasor_features.data.shape == (1, 256, 256)
    assert isinstance(phasor_features.features, pd.DataFrame)
    assert phasor_features.features.shape == (131072, 6)
    expected_columns = [
        "label",
        "G_original",
        "S_original",
        "G",
        "S",
        "harmonic",
    ]
    actual_columns = phasor_features.features.columns.tolist()
    assert actual_columns == expected_columns
    assert np.unique(phasor_features.features["harmonic"]).tolist() == [1, 2]
    # Second Channel
    layer_data_tuple = layer_data_list[1]
    assert isinstance(layer_data_tuple, tuple) and len(layer_data_tuple) == 2
    assert isinstance(layer_data_tuple[0], np.ndarray) and isinstance(
        layer_data_tuple[1], dict
    )
    assert layer_data_tuple[0].shape == (1, 256, 256)
    assert "name" in layer_data_tuple[1] and "metadata" in layer_data_tuple[1]
    assert (
        layer_data_tuple[1]["name"]
        == "test_file$EI0S Intensity Image: Channel 1"
    )
    assert (
        len(layer_data_tuple[1]["metadata"]) == 3
        and "phasor_features_labels_layer" in layer_data_tuple[1]["metadata"]
        and "original_mean" in layer_data_tuple[1]["metadata"]
        and "settings" in layer_data_tuple[1]["metadata"]
    )
    phasor_features = layer_data_tuple[1]["metadata"][
        "phasor_features_labels_layer"
    ]
    assert isinstance(phasor_features, Labels)
    assert phasor_features.data.shape == (1, 256, 256)
    assert isinstance(phasor_features.features, pd.DataFrame)
    assert phasor_features.features.shape == (131072, 6)
    expected_columns = [
        "label",
        "G_original",
        "S_original",
        "G",
        "S",
        "harmonic",
    ]
    actual_columns = phasor_features.features.columns.tolist()
    assert actual_columns == expected_columns
    assert np.unique(phasor_features.features["harmonic"]).tolist() == [1, 2]


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
    assert (
        len(layer_data_tuple[1]["metadata"]) == 3
        and "phasor_features_labels_layer" in layer_data_tuple[1]["metadata"]
        and "original_mean" in layer_data_tuple[1]["metadata"]
        and "settings" in layer_data_tuple[1]["metadata"]
    )
    phasor_features = layer_data_tuple[1]["metadata"][
        "phasor_features_labels_layer"
    ]
    assert isinstance(phasor_features, Labels)
    assert phasor_features.data.shape == (512, 512)
    assert isinstance(phasor_features.features, pd.DataFrame)
    assert phasor_features.features.shape == (524288, 6)
    expected_columns = [
        "label",
        "G_original",
        "S_original",
        "G",
        "S",
        "harmonic",
    ]
    actual_columns = phasor_features.features.columns.tolist()
    assert actual_columns == expected_columns
    assert np.unique(phasor_features.features["harmonic"]).tolist() == [1, 2]


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
    assert (
        len(layer_data_tuple[1]["metadata"]) == 3
        and "phasor_features_labels_layer" in layer_data_tuple[1]["metadata"]
        and "original_mean" in layer_data_tuple[1]["metadata"]
        and "settings" in layer_data_tuple[1]["metadata"]
    )
    phasor_features = layer_data_tuple[1]["metadata"][
        "phasor_features_labels_layer"
    ]
    assert isinstance(phasor_features, Labels)
    assert phasor_features.data.shape == (512, 512)
    assert isinstance(phasor_features.features, pd.DataFrame)
    assert phasor_features.features.shape == (524288, 6)
    expected_columns = [
        "label",
        "G_original",
        "S_original",
        "G",
        "S",
        "harmonic",
    ]
    actual_columns = phasor_features.features.columns.tolist()
    assert actual_columns == expected_columns
    assert np.unique(phasor_features.features["harmonic"]).tolist() == [1, 2]


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
    assert (
        len(layer_data_tuple[1]["metadata"]) == 3
        and "phasor_features_labels_layer" in layer_data_tuple[1]["metadata"]
        and "settings" in layer_data_tuple[1]["metadata"]
        and "original_mean" in layer_data_tuple[1]["metadata"]
    )
    phasor_features = layer_data_tuple[1]["metadata"][
        "phasor_features_labels_layer"
    ]
    assert isinstance(phasor_features, Labels)
    assert phasor_features.data.shape == (512, 512)
    assert isinstance(phasor_features.features, pd.DataFrame)
    assert phasor_features.features.shape == (524288, 6)
    expected_columns = [
        "label",
        "G_original",
        "S_original",
        "G",
        "S",
        "harmonic",
    ]
    actual_columns = phasor_features.features.columns.tolist()
    assert actual_columns == expected_columns
    assert np.unique(phasor_features.features["harmonic"]).tolist() == [1, 2]


# TODO: Add tests for .tif files
# TODO: test filter and threshold when reading OME-TIF
