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


def test_write_read_ometif_with_circular_cursors(tmp_path):
    """Test writing and reading OME-TIFF with circular cursor metadata."""
    time_constants = [0.1, 1, 10]
    raw_flim_data = make_raw_flim_data(time_constants=time_constants)
    harmonic = [1, 2, 3]
    intensity_image_layer = make_intensity_layer_with_phasors(
        raw_flim_data, harmonic=harmonic
    )

    # Add circular cursor data to metadata
    circular_cursors = [
        {'g': 0.5, 's': 0.3, 'radius': 0.1, 'color': (255, 0, 0, 255)},
        {'g': 0.6, 's': 0.4, 'radius': 0.15, 'color': (0, 255, 0, 255)},
        {'g': 0.7, 's': 0.2, 'radius': 0.08, 'color': (0, 0, 255, 255)},
    ]

    # Initialize settings structure if needed
    if "settings" not in intensity_image_layer.metadata:
        intensity_image_layer.metadata["settings"] = {}
    if "selections" not in intensity_image_layer.metadata["settings"]:
        intensity_image_layer.metadata["settings"]["selections"] = {}

    intensity_image_layer.metadata["settings"]["selections"][
        "circular_cursors"
    ] = circular_cursors

    # Write the file
    filepath = os.path.join(tmp_path, "test_cursors.ome.tif")
    write_ome_tiff(
        filepath,
        [
            (
                intensity_image_layer.data,
                {"metadata": intensity_image_layer.metadata},
            )
        ],
    )

    assert os.path.exists(filepath)

    # Read the file back
    reader = napari_get_reader(filepath, harmonics=harmonic)
    layer_data_list = reader(filepath)
    layer_data_tuple = layer_data_list[0]

    # Verify metadata was preserved
    metadata = layer_data_tuple[1]["metadata"]
    assert "settings" in metadata
    assert "selections" in metadata["settings"]
    assert "circular_cursors" in metadata["settings"]["selections"]

    # Verify circular cursor data
    restored_cursors = metadata["settings"]["selections"]["circular_cursors"]
    assert len(restored_cursors) == 3

    # Verify first cursor
    assert restored_cursors[0]['g'] == 0.5
    assert restored_cursors[0]['s'] == 0.3
    assert restored_cursors[0]['radius'] == 0.1
    assert tuple(restored_cursors[0]['color']) == (255, 0, 0, 255)

    # Verify second cursor
    assert restored_cursors[1]['g'] == 0.6
    assert restored_cursors[1]['s'] == 0.4
    assert restored_cursors[1]['radius'] == 0.15
    assert tuple(restored_cursors[1]['color']) == (0, 255, 0, 255)

    # Verify third cursor
    assert restored_cursors[2]['g'] == 0.7
    assert restored_cursors[2]['s'] == 0.2
    assert restored_cursors[2]['radius'] == 0.08
    assert tuple(restored_cursors[2]['color']) == (0, 0, 255, 255)


def test_write_ometif_without_circular_cursors(tmp_path):
    """Test writing OME-TIFF without circular cursors doesn't crash."""
    time_constants = [0.1, 1, 10]
    raw_flim_data = make_raw_flim_data(time_constants=time_constants)
    harmonic = [1, 2, 3]
    intensity_image_layer = make_intensity_layer_with_phasors(
        raw_flim_data, harmonic=harmonic
    )

    # Explicitly ensure no circular cursors in metadata
    if "settings" in intensity_image_layer.metadata:
        if "selections" in intensity_image_layer.metadata["settings"]:
            intensity_image_layer.metadata["settings"]["selections"].pop(
                "circular_cursors", None
            )

    # Write the file
    filepath = os.path.join(tmp_path, "test_no_cursors.ome.tif")
    write_ome_tiff(
        filepath,
        [
            (
                intensity_image_layer.data,
                {"metadata": intensity_image_layer.metadata},
            )
        ],
    )

    assert os.path.exists(filepath)

    # Read the file back
    reader = napari_get_reader(filepath, harmonics=harmonic)
    layer_data_list = reader(filepath)
    layer_data_tuple = layer_data_list[0]

    # Verify no circular cursors in metadata
    metadata = layer_data_tuple[1]["metadata"]
    if "settings" in metadata and "selections" in metadata["settings"]:
        assert "circular_cursors" not in metadata["settings"]["selections"]
