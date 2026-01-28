import numpy as np

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
    assert layer_data_tuple[0].shape == (256, 256)
    assert "name" in layer_data_tuple[1] and "metadata" in layer_data_tuple[1]
    assert (
        layer_data_tuple[1]["name"]
        == "Convallaria_$EI0S Intensity Image: Channel 0"
    )
    metadata = layer_data_tuple[1]["metadata"]
    assert "G" in metadata
    assert "S" in metadata
    assert "G_original" in metadata
    assert "S_original" in metadata
    assert "harmonics" in metadata
    assert "original_mean" in metadata
    assert "settings" in metadata
    # Check G and S are NumPy arrays with correct shape (n_harmonics, height, width)
    assert isinstance(metadata["G"], np.ndarray)
    assert isinstance(metadata["S"], np.ndarray)
    assert metadata["G"].shape == (2, 256, 256)
    assert metadata["S"].shape == (2, 256, 256)
    assert list(metadata["harmonics"]) == [1, 2]
    # Calibration
    layer_data_tuple = layer_data_list[1]
    assert isinstance(layer_data_tuple, tuple) and len(layer_data_tuple) == 2
    assert isinstance(layer_data_tuple[0], np.ndarray) and isinstance(
        layer_data_tuple[1], dict
    )
    assert layer_data_tuple[0].shape == (256, 256)
    assert "name" in layer_data_tuple[1] and "metadata" in layer_data_tuple[1]
    assert (
        layer_data_tuple[1]["name"]
        == "Calibration_Rhodamine110_$EI0S Intensity Image: Channel 0"
    )
    metadata = layer_data_tuple[1]["metadata"]
    assert "G" in metadata
    assert "S" in metadata
    assert "G_original" in metadata
    assert "S_original" in metadata
    assert "harmonics" in metadata
    assert "original_mean" in metadata
    assert "settings" in metadata
    # Check G and S are NumPy arrays with correct shape (n_harmonics, height, width)
    assert isinstance(metadata["G"], np.ndarray)
    assert isinstance(metadata["S"], np.ndarray)
    assert metadata["G"].shape == (2, 256, 256)
    assert metadata["S"].shape == (2, 256, 256)
    assert list(metadata["harmonics"]) == [1, 2]


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
    metadata = layer_data_tuple[1]["metadata"]
    assert "G" in metadata
    assert "S" in metadata
    assert "G_original" in metadata
    assert "S_original" in metadata
    assert "harmonics" in metadata
    assert "original_mean" in metadata
    assert "settings" in metadata
    # Check G and S are NumPy arrays with correct shape (n_harmonics, height, width)
    assert isinstance(metadata["G"], np.ndarray)
    assert isinstance(metadata["S"], np.ndarray)
    assert metadata["G"].shape == (2, 512, 512)
    assert metadata["S"].shape == (2, 512, 512)
    assert list(metadata["harmonics"]) == [1, 2]
    # Calibration
    layer_data_tuple = layer_data_list[1]
    assert isinstance(layer_data_tuple, tuple) and len(layer_data_tuple) == 2
    assert isinstance(layer_data_tuple[0], np.ndarray) and isinstance(
        layer_data_tuple[1], dict
    )
    assert layer_data_tuple[0].shape == (512, 512)
    assert "name" in layer_data_tuple[1] and "metadata" in layer_data_tuple[1]
    assert layer_data_tuple[1]["name"] == "Fluorescein_Embryo Intensity Image"
    metadata = layer_data_tuple[1]["metadata"]
    assert "G" in metadata
    assert "S" in metadata
    assert "G_original" in metadata
    assert "S_original" in metadata
    assert "harmonics" in metadata
    assert "original_mean" in metadata
    assert "settings" in metadata
    # Check G and S are NumPy arrays with correct shape (n_harmonics, height, width)
    assert isinstance(metadata["G"], np.ndarray)
    assert isinstance(metadata["S"], np.ndarray)
    assert metadata["G"].shape == (2, 512, 512)
    assert metadata["S"].shape == (2, 512, 512)
    assert list(metadata["harmonics"]) == [1, 2]


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
    metadata = layer_data_tuple[1]["metadata"]
    assert "G" in metadata
    assert "S" in metadata
    assert "G_original" in metadata
    assert "S_original" in metadata
    assert "harmonics" in metadata
    assert "original_mean" in metadata
    assert "settings" in metadata
    # Check G and S are NumPy arrays with correct shape (n_harmonics, height, width)
    assert isinstance(metadata["G"], np.ndarray)
    assert isinstance(metadata["S"], np.ndarray)
    assert metadata["G"].shape == (2, 512, 512)
    assert metadata["S"].shape == (2, 512, 512)
    assert list(metadata["harmonics"]) == [1, 2]
