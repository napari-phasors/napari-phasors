# %%
import importlib.metadata
import os

import numpy as np

from napari_phasors._reader import napari_get_reader
from napari_phasors._synthetic_generator import (
    make_intensity_layer_with_phasors,
    make_raw_flim_data,
)
from napari_phasors._writer import _convert_numpy_types, write_ome_tiff


def test_convert_numpy_types_scalars():
    """Test that numpy scalars are converted to native Python types."""
    assert _convert_numpy_types(np.int64(42)) == 42
    assert type(_convert_numpy_types(np.int64(42))) is int
    assert _convert_numpy_types(np.float64(3.14)) == 3.14
    assert type(_convert_numpy_types(np.float64(3.14))) is float
    assert _convert_numpy_types(np.bool_(True)) is True
    assert type(_convert_numpy_types(np.bool_(True))) is bool


def test_convert_numpy_types_dict_keys():
    """Test that numpy types in dict keys are converted (Issue #178)."""
    data = {np.int64(1): {'real': np.float64(0.5), 'imag': np.float64(0.3)}}
    result = _convert_numpy_types(data)
    assert result == {1: {'real': 0.5, 'imag': 0.3}}
    for key in result:
        assert type(key) is int
    for val in result[1].values():
        assert type(val) is float


def test_convert_numpy_types_nested():
    """Test recursive conversion of complex nested structures."""
    data = {
        'harmonics': np.array([1, 2, 3]),
        'positions': {
            np.int64(1): [np.float64(0.1), np.float64(0.2)],
            np.int64(2): (np.float64(0.3), np.float64(0.4)),
        },
        'plain': 'string_value',
        'count': np.int32(10),
    }
    result = _convert_numpy_types(data)
    assert result['harmonics'] == [1, 2, 3]
    assert 1 in result['positions']
    assert 2 in result['positions']
    assert result['positions'][1] == [0.1, 0.2]
    assert result['positions'][2] == (0.3, 0.4)
    assert result['plain'] == 'string_value'
    assert result['count'] == 10
    assert type(result['count']) is int


def test_convert_numpy_types_passthrough():
    """Test that non-numpy types pass through unchanged."""
    assert _convert_numpy_types('hello') == 'hello'
    assert _convert_numpy_types(42) == 42
    assert _convert_numpy_types(None) is None


def test_write_ometif_with_numpy_settings(tmp_path):
    """Test that OME-TIFF export works when settings contain numpy types.

    Regression test for Issue #178: TypeError with numpy.int64 dict keys.
    """
    time_constants = [0.1, 1, 10]
    raw_flim_data = make_raw_flim_data(time_constants=time_constants)
    harmonic = [1, 2, 3]
    intensity_image_layer = make_intensity_layer_with_phasors(
        raw_flim_data, harmonic=harmonic
    )

    # Simulate FRET tab storing numpy-typed keys and values in settings
    if "settings" not in intensity_image_layer.metadata:
        intensity_image_layer.metadata["settings"] = {}
    intensity_image_layer.metadata["settings"]["fret"] = {
        "background_positions_by_harmonic": {
            np.int64(1): {"real": np.float64(0.5), "imag": np.float64(0.3)},
            np.int64(2): {"real": np.float64(0.6), "imag": np.float64(0.4)},
        },
        "donor_lifetime": np.float64(4.0),
        "frequency": np.float64(80.0),
    }

    filepath = os.path.join(tmp_path, "test_numpy_settings.ome.tif")

    # This should NOT raise TypeError
    result = write_ome_tiff(
        filepath,
        [
            (
                intensity_image_layer.data,
                {"metadata": intensity_image_layer.metadata},
            )
        ],
    )

    assert os.path.exists(filepath)
    assert result == [filepath]

    # Read back and verify settings survived the roundtrip
    reader = napari_get_reader(filepath, harmonics=harmonic)
    layer_data_list = reader(filepath)
    metadata = layer_data_list[0][1]["metadata"]
    fret = metadata["settings"]["fret"]
    assert fret["donor_lifetime"] == 4.0
    assert fret["frequency"] == 80.0
    positions = fret["background_positions_by_harmonic"]
    # JSON keys become strings, so verify they're accessible
    assert "1" in positions or 1 in positions


def test_write_ometif_group_metadata_roundtrip(tmp_path):
    """Group metadata stored in settings['group'] survives an OME-TIFF roundtrip.

    This verifies the end-to-end OME-TIFF persistence path:
    1. A layer has group info in metadata['settings']['group'].
    2. write_ome_tiff serialises that settings dict into the file.
    3. The reader deserialises it back and the group entry is intact.
    """
    time_constants = [0.1, 1, 10]
    raw_flim_data = make_raw_flim_data(time_constants=time_constants)
    harmonic = [1, 2, 3]
    layer = make_intensity_layer_with_phasors(raw_flim_data, harmonic=harmonic)

    if "settings" not in layer.metadata:
        layer.metadata["settings"] = {}
    layer.metadata["settings"]["group"] = {
        "name": "Control",
        "color": [1.0, 0.0, 0.0],
    }

    filepath = str(tmp_path / "group_roundtrip.ome.tif")
    write_ome_tiff(
        filepath,
        [(layer.data, {"metadata": layer.metadata})],
    )

    assert os.path.exists(filepath)

    reader = napari_get_reader(filepath, harmonics=harmonic)
    layer_data_list = reader(filepath)
    restored_settings = layer_data_list[0][1]["metadata"]["settings"]

    assert "group" in restored_settings
    grp = restored_settings["group"]
    assert grp["name"] == "Control"
    assert grp["color"] == [1.0, 0.0, 0.0]


def test_write_ometif_group_metadata_with_colormap_roundtrip(tmp_path):
    """Group metadata that includes a colormap/style entry (from the contour
    dialog) also survives an OME-TIFF roundtrip."""
    time_constants = [0.1, 1, 10]
    raw_flim_data = make_raw_flim_data(time_constants=time_constants)
    harmonic = [1]
    layer = make_intensity_layer_with_phasors(raw_flim_data, harmonic=harmonic)

    if "settings" not in layer.metadata:
        layer.metadata["settings"] = {}
    layer.metadata["settings"]["group"] = {
        "name": "Treated",
        "color": [0.0, 0.5, 1.0],
        "style": "colormap",
        "colormap": "plasma",
    }

    filepath = str(tmp_path / "group_colormap_roundtrip.ome.tif")
    write_ome_tiff(
        filepath,
        [(layer.data, {"metadata": layer.metadata})],
    )

    reader = napari_get_reader(filepath, harmonics=harmonic)
    layer_data_list = reader(filepath)
    grp = layer_data_list[0][1]["metadata"]["settings"]["group"]

    assert grp["name"] == "Treated"
    assert grp["style"] == "colormap"
    assert grp["colormap"] == "plasma"
    assert grp["color"] == [0.0, 0.5, 1.0]


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
    write_ome_tiff(
        os.path.join(tmp_path, "test_file_extension.ome.tiff"),
        [
            (
                intensity_image_layer.data,
                {"metadata": intensity_image_layer.metadata},
            )
        ],
    )
    assert os.path.exists(
        os.path.join(tmp_path, "test_file_extension.ome.tiff")
    )
    assert not os.path.exists(
        os.path.join(tmp_path, "test_file_extension.ome.tiff.ome.tif")
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
    if (
        "settings" in intensity_image_layer.metadata
        and "selections" in intensity_image_layer.metadata["settings"]
    ):
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


def test_write_ometif_without_phasor_data(tmp_path):
    """Test writing OME-TIFF files for layers without phasor data."""
    import tifffile
    from napari.layers import Image

    # Create a simple image layer without phasor data
    data = np.random.random((100, 100))
    layer = Image(data, name="test_image")

    # Add some metadata but no phasor data
    layer.metadata = {
        "some_info": "test",
        "values": [1, 2, 3],
    }

    # Write the file
    filepath = os.path.join(tmp_path, "test_no_phasor.ome.tif")
    write_ome_tiff(filepath, layer)

    assert os.path.exists(filepath)

    # Read the file back and verify it contains the data
    with tifffile.TiffFile(filepath) as tif:
        loaded_data = tif.asarray()
        np.testing.assert_array_almost_equal(loaded_data, data)

        # Check that metadata was saved
        if tif.pages[0].description:
            import json

            description = json.loads(tif.pages[0].description)
            assert "napari_phasors_settings" in description


def test_write_ometif_saves_z_spacing_for_3d_layer(tmp_path):
    """Save z-spacing metadata only when a Z axis is present."""
    import json

    import tifffile
    from napari.layers import Image

    data = np.random.random((4, 10, 12))
    layer = Image(data, name="z_stack")
    layer.scale = (2.75, 1.0, 1.0)
    layer.metadata = {"settings": {}}

    filepath = os.path.join(tmp_path, "test_z_spacing_3d.ome.tif")
    write_ome_tiff(filepath, layer)

    with tifffile.TiffFile(filepath) as tif:
        description = json.loads(tif.pages[0].description)
        settings = json.loads(description["napari_phasors_settings"])
        assert settings["z_spacing_um"] == 2.75


def test_write_ometif_does_not_save_z_spacing_for_2d_layer(tmp_path):
    """Do not save z-spacing metadata for 2D layers."""
    import json

    import tifffile
    from napari.layers import Image

    data = np.random.random((10, 12))
    layer = Image(data, name="image_2d")
    layer.scale = (3.5, 1.0)
    layer.metadata = {"settings": {}}

    filepath = os.path.join(tmp_path, "test_z_spacing_2d.ome.tif")
    write_ome_tiff(filepath, layer)

    with tifffile.TiffFile(filepath) as tif:
        description = json.loads(tif.pages[0].description)
        settings = json.loads(description["napari_phasors_settings"])
        assert "z_spacing_um" not in settings


def test_write_ometif_masked(tmp_path):
    """Test that write_ome_tiff with export_masked=True applies the mask."""
    time_constants = [0.1, 1, 10]
    raw_flim_data = make_raw_flim_data(time_constants=time_constants)
    harmonic = [1, 2, 3]
    intensity_image_layer = make_intensity_layer_with_phasors(
        raw_flim_data, harmonic=harmonic
    )

    mask = np.ones((2, 5), dtype=int)
    mask[0, 0] = 0
    intensity_image_layer.metadata["mask"] = mask
    intensity_image_layer.metadata["mask_invert"] = False

    # 1. Export with export_masked=True
    filepath_masked = os.path.join(tmp_path, "test_masked.ome.tif")
    write_ome_tiff(
        filepath_masked,
        [
            (
                intensity_image_layer.data,
                {"metadata": intensity_image_layer.metadata},
            )
        ],
        export_masked=True,
    )

    assert os.path.exists(filepath_masked)

    # Read back and verify G, S, and mean have NaN at [0,0]
    reader = napari_get_reader(filepath_masked, harmonics=harmonic)
    layer_data_list = reader(filepath_masked)
    metadata_masked = layer_data_list[0][1]["metadata"]
    mean_masked = layer_data_list[0][0]

    assert np.isnan(mean_masked[0, 0])
    assert np.isnan(metadata_masked["G"][:, 0, 0]).all()
    assert np.isnan(metadata_masked["S"][:, 0, 0]).all()

    # The rest should not be NaN
    assert not np.isnan(mean_masked[0, 1:]).any()
    assert not np.isnan(metadata_masked["G"][:, 0, 1:]).any()

    # 2. Export with export_masked=False
    filepath_unmasked = os.path.join(tmp_path, "test_unmasked.ome.tif")
    write_ome_tiff(
        filepath_unmasked,
        [
            (
                intensity_image_layer.data,
                {"metadata": intensity_image_layer.metadata},
            )
        ],
        export_masked=False,
    )

    reader_unmasked = napari_get_reader(filepath_unmasked, harmonics=harmonic)
    layer_data_list_unmasked = reader_unmasked(filepath_unmasked)
    metadata_unmasked = layer_data_list_unmasked[0][1]["metadata"]
    mean_unmasked = layer_data_list_unmasked[0][0]

    assert not np.isnan(mean_unmasked[0, 0])
    assert not np.isnan(metadata_unmasked["G"][:, 0, 0]).any()


def test_write_ometif_masked_phasor_same_ndim(tmp_path):
    """Test write_ome_tiff with export_masked=True, has_phasor_data=True, and G.ndim == mask.ndim."""
    from unittest.mock import patch

    mean = np.ones((2, 5))
    G = np.ones((2, 5))
    S = np.ones((2, 5))
    mask = np.ones((2, 5), dtype=int)
    mask[0, 0] = 0  # invalid

    metadata = {
        "original_mean": mean,
        "G_original": G,
        "S_original": S,
        "harmonics": [1],
        "mask": mask,
        "mask_invert": False,
    }

    filepath = os.path.join(tmp_path, "test_same_ndim.ome.tif")

    with patch(
        "napari_phasors._writer.phasor_to_ometiff"
    ) as mock_phasor_to_ometiff:
        write_ome_tiff(
            filepath,
            [(mean, {"metadata": metadata})],
            export_masked=True,
        )

        mock_phasor_to_ometiff.assert_called_once()
        args, kwargs = mock_phasor_to_ometiff.call_args

        called_mean = args[1]
        called_G = args[2]
        called_S = args[3]

        assert np.isnan(called_mean[0, 0])
        assert np.isnan(called_G[0, 0])
        assert np.isnan(called_S[0, 0])

        assert not np.isnan(called_mean[0, 1:]).any()
        assert not np.isnan(called_G[0, 1:]).any()
        assert not np.isnan(called_S[0, 1:]).any()


def test_write_ometif_masked_non_phasor(tmp_path):
    """Test write_ome_tiff with export_masked=True for non-phasor layers."""
    from unittest.mock import patch

    from napari.layers import Image

    # Case 1: data.ndim > mask_invalid.ndim
    data_3d = np.ones((3, 2, 5))
    mask_2d = np.ones((2, 5), dtype=int)
    mask_2d[0, 0] = 0  # invalid

    layer_3d = Image(data_3d, name="layer_3d")
    layer_3d.metadata = {
        "mask": mask_2d,
        "mask_invert": False,
    }

    filepath_3d = os.path.join(tmp_path, "test_non_phasor_3d.ome.tif")

    with patch("tifffile.imwrite") as mock_imwrite:
        write_ome_tiff(
            filepath_3d,
            layer_3d,
            export_masked=True,
        )

        mock_imwrite.assert_called_once()
        args, kwargs = mock_imwrite.call_args
        written_data = args[1]

        assert written_data.shape == (3, 2, 5)
        assert np.isnan(written_data[:, 0, 0]).all()
        assert not np.isnan(written_data[:, 0, 1:]).any()

    # Case 2: data.ndim == mask_invalid.ndim with mask_invert = True
    data_2d = np.ones((2, 5))
    mask_2d_invert = np.zeros((2, 5), dtype=int)
    mask_2d_invert[0, 0] = 1  # invalid when invert=True

    layer_2d = Image(data_2d, name="layer_2d")
    layer_2d.metadata = {
        "mask": mask_2d_invert,
        "mask_invert": True,
    }

    filepath_2d = os.path.join(tmp_path, "test_non_phasor_2d.ome.tif")

    with patch("tifffile.imwrite") as mock_imwrite:
        write_ome_tiff(
            filepath_2d,
            layer_2d,
            export_masked=True,
        )

        mock_imwrite.assert_called_once()
        args, kwargs = mock_imwrite.call_args
        written_data = args[1]

        assert written_data.shape == (2, 5)
        assert np.isnan(written_data[0, 0])
        assert not np.isnan(written_data[0, 1:]).any()


def test_export_layer_as_image_tuple_colormap(tmp_path):
    """Test export_layer_as_image with a layer data tuple containing a dict colormap."""
    from PIL import Image as PILImage

    from napari_phasors._writer import export_layer_as_image

    # Use a gradient data array to ensure we span the colormap
    data = np.linspace(0, 1, 100).reshape((10, 10))

    # A custom colormap dict with different red and blue endpoints
    colormap_dict = {
        "name": "custom_red_blue",
        "colors": [
            [1.0, 0.0, 0.0, 1.0],  # Red at 0
            [0.0, 0.0, 1.0, 1.0],  # Blue at 1
        ],
    }

    layer_tuple = (
        data,
        {
            "name": "test_image",
            "colormap": colormap_dict,
            "contrast_limits": [0.0, 1.0],
            "metadata": {},
        },
        "image",
    )

    export_path = os.path.join(tmp_path, "test_image_export.png")
    export_layer_as_image(export_path, layer_tuple, include_colorbar=False)

    assert os.path.exists(export_path)

    # Open and verify the image has color representation (not black and white)
    with PILImage.open(export_path) as img:
        img_data = np.array(img)

    assert img_data.ndim == 3
    assert img_data.shape[-1] in (3, 4)

    # Verify that the image is colored (R != B, since some parts are red, some blue)
    is_colored = np.any(img_data[..., 0] != img_data[..., 2])
    assert is_colored, "Exported image is grayscale!"


# ---------------------------------------------------------------------------
# export_layer_as_csv
# ---------------------------------------------------------------------------


def test_export_csv_phasor_multiharmonic(tmp_path):
    """CSV export of a phasor layer with multiple harmonics (3D G/S)."""
    import pandas as pd
    from napari.layers import Image

    from napari_phasors._writer import export_layer_as_csv

    mean = np.ones((4, 4))
    rng = np.random.default_rng(0)
    G = rng.random((2, 4, 4))
    S = rng.random((2, 4, 4))
    # A fully-NaN pixel must be skipped in the output.
    G[:, 0, 0] = np.nan
    S[:, 0, 0] = np.nan
    layer = Image(
        mean,
        name="phasor_img",
        metadata={
            "G": G,
            "S": S,
            "G_original": G.copy(),
            "S_original": S.copy(),
            "harmonics": [1, 2],
        },
    )
    out = export_layer_as_csv(str(tmp_path / "out.csv"), layer)
    assert len(out) == 1 and os.path.exists(out[0])
    df = pd.read_csv(out[0])
    assert {
        "harmonic",
        "G",
        "S",
        "G_original",
        "S_original",
        "dim_0",
        "dim_1",
    }.issubset(df.columns)
    # 2 harmonics * (16 - 1 NaN pixel) rows.
    assert len(df) == 2 * 15


def test_export_csv_phasor_single_harmonic_2d(tmp_path):
    """CSV export with 2D G/S arrays (single harmonic)."""
    import pandas as pd
    from napari.layers import Image

    from napari_phasors._writer import export_layer_as_csv

    G = np.random.default_rng(1).random((4, 4))
    S = np.random.default_rng(2).random((4, 4))
    layer = Image(
        np.ones((4, 4)),
        name="phasor2d",
        metadata={"G": G, "S": S, "harmonics": 1},
    )
    out = export_layer_as_csv(str(tmp_path / "out2d.csv"), layer)
    df = pd.read_csv(out[0])
    assert len(df) == 16
    assert list(df["harmonic"].unique()) == [1]


def test_export_csv_no_phasor_2d(tmp_path):
    """CSV export of a plain 2D layer produces y/x/value columns."""
    import pandas as pd
    from napari.layers import Image

    from napari_phasors._writer import export_layer_as_csv

    layer = Image(np.arange(16).reshape(4, 4), name="raw2d")
    out = export_layer_as_csv(str(tmp_path / "raw2d.csv"), layer)
    df = pd.read_csv(out[0])
    assert list(df.columns) == ["y", "x", "value"]
    assert len(df) == 16


def test_export_csv_no_phasor_nd_tuple(tmp_path):
    """CSV export of an N-D (>2) layer-data tuple uses dim_* columns."""
    import pandas as pd

    from napari_phasors._writer import export_layer_as_csv

    data = np.arange(24).reshape(2, 3, 4)
    layer_tuple = (data, {"name": "vol", "metadata": {}})
    out = export_layer_as_csv(str(tmp_path / "vol.csv"), layer_tuple)
    df = pd.read_csv(out[0])
    assert {"dim_0", "dim_1", "dim_2", "value"}.issubset(df.columns)
    assert len(df) == 24


# ---------------------------------------------------------------------------
# export_layer_as_image — additional branches
# ---------------------------------------------------------------------------


def test_export_image_labels_layer(tmp_path):
    """Labels layers are mapped through their colormap and exported."""
    from napari.layers import Labels

    from napari_phasors._writer import export_layer_as_image

    labels = Labels(np.array([[0, 1, 2], [3, 0, 1]], dtype=int), name="lbls")
    out = export_layer_as_image(
        str(tmp_path / "lbls.png"), labels, include_colorbar=True
    )
    assert os.path.exists(out[0])


def test_export_image_colormap_object_with_colorbar(tmp_path):
    """A napari Colormap object (with .colors) and a colorbar are handled."""
    from napari.layers import Image

    from napari_phasors._writer import export_layer_as_image

    img = Image(
        np.linspace(0, 1, 16).reshape(4, 4),
        name="vir",
        colormap="viridis",
    )
    out = export_layer_as_image(
        str(tmp_path / "vir.png"), img, include_colorbar=True
    )
    assert os.path.exists(out[0])


def test_export_image_jpeg_output(tmp_path):
    """JPEG export switches the figure to a white facecolor."""
    from napari.layers import Image

    from napari_phasors._writer import export_layer_as_image

    img = Image(np.linspace(0, 1, 16).reshape(4, 4), name="g", colormap="gray")
    out = export_layer_as_image(
        str(tmp_path / "g.jpg"), img, include_colorbar=True
    )
    assert os.path.exists(out[0])


def test_export_image_multidim_uses_current_step(tmp_path):
    """Multi-dimensional layers slice according to current_step."""
    from napari.layers import Image

    from napari_phasors._writer import export_layer_as_image

    img = Image(
        np.random.default_rng(3).random((3, 4, 4)),
        name="stack",
        colormap="gray",
    )
    out = export_layer_as_image(
        str(tmp_path / "stack.png"),
        img,
        current_step=[1, 0, 0],
        include_colorbar=False,
    )
    assert os.path.exists(out[0])


def test_export_image_gamma_applies_power_norm(tmp_path):
    """A layer with gamma != 1.0 is exported via a PowerNorm, not plain vmin/vmax.

    Regression: previously the exported image used the raw contrast limits
    and ignored gamma entirely, so it didn't match napari's on-screen
    rendering whenever gamma correction was applied.
    """
    from napari.layers import Image
    from PIL import Image as PILImage

    from napari_phasors._writer import export_layer_as_image

    data = np.linspace(0, 1, 16).reshape(4, 4)

    linear_layer = Image(data.copy(), name="linear", colormap="gray")
    linear_layer.gamma = 1.0
    gamma_layer = Image(data.copy(), name="gamma", colormap="gray")
    gamma_layer.gamma = 2.5

    linear_path = export_layer_as_image(
        str(tmp_path / "linear.png"), linear_layer, include_colorbar=False
    )[0]
    gamma_path = export_layer_as_image(
        str(tmp_path / "gamma.png"), gamma_layer, include_colorbar=False
    )[0]

    with PILImage.open(linear_path) as linear_img:
        linear_arr = np.array(linear_img)
    with PILImage.open(gamma_path) as gamma_img:
        gamma_arr = np.array(gamma_img)

    assert not np.array_equal(linear_arr, gamma_arr)


# ---------------------------------------------------------------------------
# write_ome_tiff — dims / physical-size branches
# ---------------------------------------------------------------------------


def test_write_ometif_4d_and_5d_raw_dims(tmp_path):
    """Raw (non-phasor) layers get TZYX/TZCYX dims for 4D/5D data."""

    import tifffile
    from napari.layers import Image

    for ndim, shape, expected in (
        (4, (2, 2, 4, 4), "TZYX"),
        (5, (2, 2, 2, 4, 4), "TZCYX"),
    ):
        layer = Image(np.ones(shape), name=f"vol{ndim}")
        paths = write_ome_tiff(str(tmp_path / f"vol{ndim}.ome.tif"), layer)
        assert os.path.exists(paths[0])
        with tifffile.TiffFile(paths[0]) as tif:
            axes = tif.series[0].axes
        assert axes == expected


def test_write_ometif_physical_size_from_scale(tmp_path):
    """Layer scale and Y/X axis labels populate PhysicalSizeX/Y in OME-XML."""
    import tifffile
    from napari.layers import Image

    mean = np.ones((4, 4))
    G = np.zeros((1, 4, 4))
    S = np.zeros((1, 4, 4))
    layer = Image(
        mean,
        name="scaled",
        scale=(2.0, 3.0),
        metadata={
            "original_mean": mean.copy(),
            "G_original": G,
            "S_original": S,
            "harmonics": [1],
        },
    )
    layer.axis_labels = ("y", "x")
    paths = write_ome_tiff(str(tmp_path / "scaled.ome.tif"), layer)
    assert os.path.exists(paths[0])
    with tifffile.TiffFile(paths[0]) as tif:
        meta = tif.ome_metadata or ""
    assert "PhysicalSizeY" in meta and "PhysicalSizeX" in meta


def _make_phasor_3d_layer(scale, settings=None):
    from napari.layers import Image

    mean = np.ones((3, 4, 4))
    G = np.zeros((1, 3, 4, 4))
    S = np.zeros((1, 3, 4, 4))
    metadata = {
        "original_mean": mean.copy(),
        "G_original": G,
        "S_original": S,
        "harmonics": [1],
    }
    if settings is not None:
        metadata["settings"] = settings
    layer = Image(mean, name="zstack", scale=scale, metadata=metadata)
    layer.axis_labels = ("z", "y", "x")
    return layer


def test_write_ometif_z_spacing_from_z_axis_label(tmp_path):
    """A 'z' axis label with a positive scale sets PhysicalSizeZ."""
    import tifffile

    layer = _make_phasor_3d_layer(scale=(5.0, 1.0, 1.0))
    paths = write_ome_tiff(str(tmp_path / "zlabel.ome.tif"), layer)
    with tifffile.TiffFile(paths[0]) as tif:
        meta = tif.ome_metadata or ""
    assert "PhysicalSizeZ" in meta


def test_write_ometif_z_spacing_falls_back_to_settings(tmp_path):
    """A layer-data tuple (no scale attr) uses z_spacing_um from settings."""
    import tifffile

    mean = np.ones((3, 4, 4))
    G = np.zeros((1, 3, 4, 4))
    layer_tuple = (
        mean,
        {
            "name": "zsettings",
            "metadata": {
                "original_mean": mean.copy(),
                "G_original": G,
                "S_original": G.copy(),
                "harmonics": [1],
                "settings": {"z_spacing_um": 4.0},
            },
        },
    )
    paths = write_ome_tiff(str(tmp_path / "zsettings.ome.tif"), layer_tuple)
    with tifffile.TiffFile(paths[0]) as tif:
        meta = tif.ome_metadata or ""
    assert "PhysicalSizeZ" in meta


def test_export_image_list_of_layers(tmp_path):
    """Passing a list of layers exports each one."""
    from napari.layers import Image

    from napari_phasors._writer import export_layer_as_image

    layers = [
        Image(np.linspace(0, 1, 16).reshape(4, 4), name="a", colormap="gray"),
        Image(np.linspace(0, 1, 16).reshape(4, 4), name="b", colormap="gray"),
    ]
    out = export_layer_as_image(
        str(tmp_path / "multi.png"), layers, include_colorbar=False
    )
    assert len(out) == 2 and all(os.path.exists(p) for p in out)


def test_export_csv_list_of_layers(tmp_path):
    """Passing a list of layers exports a CSV per layer."""
    from napari.layers import Image

    from napari_phasors._writer import export_layer_as_csv

    layers = [
        Image(np.arange(16).reshape(4, 4), name="a"),
        Image(np.arange(16).reshape(4, 4), name="b"),
    ]
    out = export_layer_as_csv(str(tmp_path / "multi.csv"), layers)
    assert len(out) == 2 and all(os.path.exists(p) for p in out)


def test_export_image_multidim_without_current_step(tmp_path):
    """Multi-dimensional layers without current_step default to index 0."""
    from napari.layers import Image

    from napari_phasors._writer import export_layer_as_image

    img = Image(
        np.random.default_rng(4).random((3, 4, 4)),
        name="stack",
        colormap="gray",
    )
    out = export_layer_as_image(
        str(tmp_path / "nostep.png"), img, include_colorbar=False
    )
    assert os.path.exists(out[0])


def test_export_image_dict_colormap_without_colors(tmp_path):
    """A dict colormap with colors=None falls back to a named matplotlib cmap."""
    from napari_phasors._writer import export_layer_as_image

    data = np.linspace(0, 1, 16).reshape(4, 4)
    layer_tuple = (
        data,
        {
            "name": "x",
            "colormap": {"name": "viridis", "colors": None},
            "metadata": {},
        },
        "image",
    )
    out = export_layer_as_image(
        str(tmp_path / "dictcmap.png"), layer_tuple, include_colorbar=False
    )
    assert os.path.exists(out[0])


def test_export_image_string_colormap(tmp_path):
    """A plain string colormap is resolved via matplotlib."""
    from napari_phasors._writer import export_layer_as_image

    data = np.linspace(0, 1, 16).reshape(4, 4)
    layer_tuple = (
        data,
        {"name": "x", "colormap": "plasma", "metadata": {}},
        "image",
    )
    out = export_layer_as_image(
        str(tmp_path / "strcmap.png"), layer_tuple, include_colorbar=False
    )
    assert os.path.exists(out[0])


def _phasor_image(name):
    from napari.layers import Image

    mean = np.ones((4, 4))
    G = np.zeros((1, 4, 4))
    return Image(
        mean,
        name=name,
        metadata={
            "original_mean": mean.copy(),
            "G_original": G,
            "S_original": G.copy(),
            "harmonics": [1],
        },
    )


def test_write_ometif_manual_selections_stripped(tmp_path):
    """manual_selections are removed from persisted settings on write."""
    raw_flim_data = make_raw_flim_data(time_constants=[0.1, 1, 10])
    layer = make_intensity_layer_with_phasors(raw_flim_data, harmonic=[1, 2])
    layer.metadata.setdefault("settings", {})["selections"] = {
        "manual_selections": [1, 2, 3],
        "circular_cursors": [
            {"g": 0.5, "s": 0.3, "radius": 0.1, "color": (255, 0, 0, 255)}
        ],
    }
    filepath = str(tmp_path / "sel.ome.tif")
    write_ome_tiff(filepath, layer)

    reader = napari_get_reader(filepath, harmonics=[1, 2])
    metadata = reader(filepath)[0][1]["metadata"]
    selections = metadata["settings"]["selections"]
    assert "manual_selections" not in selections
    assert "circular_cursors" in selections


def test_write_ometif_multilayer_list_naming(tmp_path):
    """Exporting a list of >1 layers disambiguates filenames per layer."""
    layers = [_phasor_image("img1"), _phasor_image("img2")]
    # Custom base name (not a layer name) -> "<base>_<layer>.ome.tif".
    paths = write_ome_tiff(str(tmp_path / "custom.ome.tif"), layers)
    assert len(paths) == 2
    names = {os.path.basename(p) for p in paths}
    assert names == {"custom_img1.ome.tif", "custom_img2.ome.tif"}


def test_write_ometif_multi_save_via_viewer(make_napari_viewer, tmp_path):
    """A single-layer call while multiple layers are selected in the viewer
    triggers the multi-save naming path."""
    viewer = make_napari_viewer()
    l1 = viewer.add_layer(_phasor_image("img1"))
    l2 = viewer.add_layer(_phasor_image("img2"))
    viewer.layers.selection = {l1, l2}
    # Custom base name -> "<base>_<layer>.ome.tif".
    paths = write_ome_tiff(str(tmp_path / "custom.ome.tif"), l1)
    assert os.path.basename(paths[0]) == "custom_img1.ome.tif"
    # Base name matching the layer name -> just "<layer>.ome.tif".
    paths = write_ome_tiff(str(tmp_path / "img1.ome.tif"), l1)
    assert os.path.basename(paths[0]) == "img1.ome.tif"
