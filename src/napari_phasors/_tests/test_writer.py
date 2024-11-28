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
    phasor_features = layer_data_tuple[1]["metadata"][
        "phasor_features_labels_layer"
    ]
    np.testing.assert_array_equal(
        phasor_features.data, [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
    )
    assert phasor_features.features.shape == (30, 6)
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
    assert phasor_features.features["harmonic"].unique().tolist() == [1, 2, 3]
