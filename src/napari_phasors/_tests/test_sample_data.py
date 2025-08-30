import numpy as np
import pandas as pd
from napari.layers import Labels

from napari_phasors._sample_data import (
    convallaria_FLIM_sample_data,
    embryo_FLIM_sample_data,
    paramecium_HSI_sample_data,
)


def test_convallaria_FLIM_sample_data(make_napari_viewer):
    """Test the convallaria FLIM sample data"""
    layer_data_list = convallaria_FLIM_sample_data()
    assert isinstance(layer_data_list, list) and len(layer_data_list) == 2
    # Convallaria image
    layer_data_tuple = layer_data_list[0]
    assert isinstance(layer_data_tuple, tuple) and len(layer_data_tuple) == 2
    assert isinstance(layer_data_tuple[0], np.ndarray) and isinstance(
        layer_data_tuple[1], dict
    )
    assert layer_data_tuple[0].shape == (1, 256, 256)
    assert "name" in layer_data_tuple[1] and "metadata" in layer_data_tuple[1]
    assert (
        layer_data_tuple[1]["name"]
        == "Convallaria_$EI0S Intensity Image: Channel 0"
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
    # Calibration
    layer_data_tuple = layer_data_list[1]
    assert isinstance(layer_data_tuple, tuple) and len(layer_data_tuple) == 2
    assert isinstance(layer_data_tuple[0], np.ndarray) and isinstance(
        layer_data_tuple[1], dict
    )
    assert layer_data_tuple[0].shape == (1, 256, 256)
    assert "name" in layer_data_tuple[1] and "metadata" in layer_data_tuple[1]
    assert (
        layer_data_tuple[1]["name"]
        == "Calibration_Rhodamine110_$EI0S Intensity Image: Channel 0"
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


def test_embryo_FLIM_sample_data(make_napari_viewer):
    """Test the embryo FLIM sample data"""
    layer_data_list = embryo_FLIM_sample_data()
    assert isinstance(layer_data_list, list) and len(layer_data_list) == 2
    # Embryo image
    layer_data_tuple = layer_data_list[0]
    assert isinstance(layer_data_tuple, tuple) and len(layer_data_tuple) == 2
    assert isinstance(layer_data_tuple[0], np.ndarray) and isinstance(
        layer_data_tuple[1], dict
    )
    assert layer_data_tuple[0].shape == (512, 512)
    assert "name" in layer_data_tuple[1] and "metadata" in layer_data_tuple[1]
    assert layer_data_tuple[1]["name"] == "Embryo Intensity Image"
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
    # Calibration
    layer_data_tuple = layer_data_list[1]
    assert isinstance(layer_data_tuple, tuple) and len(layer_data_tuple) == 2
    assert isinstance(layer_data_tuple[0], np.ndarray) and isinstance(
        layer_data_tuple[1], dict
    )
    assert layer_data_tuple[0].shape == (512, 512)
    assert "name" in layer_data_tuple[1] and "metadata" in layer_data_tuple[1]
    assert layer_data_tuple[1]["name"] == "Fluorescein_Embryo Intensity Image"
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


def test_paramecium_HSI_sample_data(make_napari_viewer):
    """Test the paramecium HSI sample data"""
    layer_data_list = paramecium_HSI_sample_data()
    assert isinstance(layer_data_list, list) and len(layer_data_list) == 1
    layer_data_tuple = layer_data_list[0]
    assert isinstance(layer_data_tuple, tuple) and len(layer_data_tuple) == 2
    assert isinstance(layer_data_tuple[0], np.ndarray) and isinstance(
        layer_data_tuple[1], dict
    )
    assert layer_data_tuple[0].shape == (512, 512)
    assert "name" in layer_data_tuple[1] and "metadata" in layer_data_tuple[1]
    assert layer_data_tuple[1]["name"] == "paramecium Intensity Image"
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
