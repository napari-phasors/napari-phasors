# %%
import importlib.metadata
import os

import numpy as np

from napari_phasors._reader import napari_get_reader
from napari_phasors._synthetic_generator import (
    make_intensity_layer_with_phasors,
    make_raw_flim_data,
)
from napari_phasors._writer import write_ome_tiff


def test_write_ometif(tmp_path):
    time_constants = [0.1, 1, 10]
    raw_flim_data = make_raw_flim_data(time_constants=time_constants)
    harmonic = [1, 2, 3]
    intensity_image_layer = make_intensity_layer_with_phasors(
        raw_flim_data, harmonic=harmonic
    )
    write_ome_tiff(
        os.path.join(tmp_path, "test_file"),
        [
            (
                intensity_image_layer.data,
                {"metadata": intensity_image_layer.metadata},
            )
        ],
    )
    assert os.path.exists(os.path.join(tmp_path, "test_file.ome.tif"))
    write_ome_tiff(
        os.path.join(tmp_path, "test_file_extension.ome.tif"),
        [
            (
                intensity_image_layer.data,
                {"metadata": intensity_image_layer.metadata},
            )
        ],
    )
    assert os.path.exists(
        os.path.join(tmp_path, "test_file_extension.ome.tif")
    )
    reader = napari_get_reader(
        os.path.join(tmp_path, "test_file.ome.tif"), harmonics=harmonic
    )
    layer_data_list = reader(os.path.join(tmp_path, "test_file.ome.tif"))
    layer_data_tuple = layer_data_list[0]
    assert len(layer_data_tuple) == 2
    np.testing.assert_array_almost_equal(
        layer_data_tuple[0], intensity_image_layer.data
    )
    assert layer_data_tuple[1]["metadata"]["settings"]["version"] == str(
        importlib.metadata.version("napari-phasors")
    )
    # Check phasor data in metadata (new array-based structure)
    metadata = layer_data_tuple[1]["metadata"]
    assert "G" in metadata
    assert "S" in metadata
    assert "G_original" in metadata
    assert "S_original" in metadata
    assert "harmonics" in metadata
    # Check harmonics
    assert list(metadata["harmonics"]) == [1, 2, 3]
    # Check G and S shapes: (n_harmonics, height, width)
    assert metadata["G"].shape == (3, 2, 5)
    assert metadata["S"].shape == (3, 2, 5)
    assert metadata["G_original"].shape == (3, 2, 5)
    assert metadata["S_original"].shape == (3, 2, 5)
