# %%
import importlib.metadata
import os

import numpy as np
import pytest

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

    # A 2D image has no signal axis: napari's own reader opens it as is.
    ((read,),) = napari_get_reader(filepath)(filepath)
    np.testing.assert_array_equal(read, data)


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
    """The mask, its invert flag, label selection and Shapes vertices are
    stored in the file and read back into the layer metadata, while the
    phasor data itself is written unmasked.
    """
    import tifffile

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
    intensity_image_layer.metadata["mask_labels"] = [np.int64(1)]
    square = np.array([[0, 1], [0, 4], [1, 4], [1, 1]], dtype=float)
    intensity_image_layer.metadata["mask_shapes"] = {
        "data": [square],
        "shape_type": ["rectangle"],
    }

    filepath_unmasked = os.path.join(tmp_path, "test_unmasked.ome.tif")
    write_ome_tiff(
        filepath_unmasked,
        [
            (
                intensity_image_layer.data,
                {"metadata": intensity_image_layer.metadata},
            )
        ],
    )

    reader_unmasked = napari_get_reader(filepath_unmasked, harmonics=harmonic)
    layer_data_list_unmasked = reader_unmasked(filepath_unmasked)
    metadata_unmasked = layer_data_list_unmasked[0][1]["metadata"]
    mean_unmasked = layer_data_list_unmasked[0][0]

    assert not np.isnan(mean_unmasked[0, 0])
    assert not np.isnan(metadata_unmasked["G"][:, 0, 0]).any()

    np.testing.assert_array_equal(metadata_unmasked["mask"], mask)
    assert metadata_unmasked["mask_invert"] is False
    assert metadata_unmasked["mask_labels"] == [1]
    assert metadata_unmasked["mask_shapes"]["shape_type"] == ["rectangle"]
    np.testing.assert_array_equal(
        metadata_unmasked["mask_shapes"]["data"][0], square
    )
    with tifffile.TiffFile(filepath_unmasked) as tif:
        assert [series.name for series in tif.series][:3] == [
            "Phasor mean",
            "Phasor real",
            "Phasor imag",
        ]

    # A damaged mask page is ignored; the phasors still load.
    with tifffile.TiffWriter(filepath_unmasked, append="force") as tif:
        tif.write(
            mask,
            description='{"napari_phasors_mask": "not a dict"}',
            metadata=None,
        )
    damaged = napari_get_reader(filepath_unmasked)(filepath_unmasked)
    assert "mask" not in damaged[0][1]["metadata"]


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


def _cursor_selection_labels(name="selection"):
    """A Labels layer coloured like the ones the cursor selections create."""
    from napari.layers import Labels
    from napari.utils import DirectLabelColormap

    return Labels(
        np.array([[0, 1, 2], [3, 0, 1]], dtype=int),
        name=name,
        colormap=DirectLabelColormap(
            color_dict={
                0: [0, 0, 0, 0],
                1: [1, 0, 0, 1],
                2: [0, 1, 0, 1],
                3: [0, 0, 1, 1],
                None: [0, 0, 0, 0],
            },
            name="manual_selection_colors",
        ),
    )


def test_export_labels_layer_as_svg(tmp_path):
    """A cursor-selection Labels layer is written as a real SVG."""
    from napari_phasors._writer import export_layer_as_image

    out = export_layer_as_image(
        str(tmp_path / "selection.svg"), _cursor_selection_labels()
    )
    assert out == [str(tmp_path / "selection.svg")]
    assert "<svg" in (tmp_path / "selection.svg").read_text()


def test_svg_writer_routing_uses_napari_phasors_for_labels_only():
    """Labels go to napari-phasors for ``.svg``; other layers keep napari-svg.

    napari-svg cannot draw a ``DirectLabelColormap``, so a cursor selection
    failed to save. The manifest claims the extension for labels only, so
    the layers it handled correctly are still handled by it.
    """
    from npe2 import PluginManager

    pm = PluginManager.instance()
    pm.discover()

    def writer_for(*layer_types):
        writer, _ = pm.get_writer("layer.svg", layer_types=list(layer_types))
        return writer.command

    assert writer_for("labels") == "napari-phasors.export_layer_as_image"
    assert (
        writer_for("labels", "labels")
        == "napari-phasors.export_layer_as_image"
    )
    assert writer_for("image") == "napari-svg.svg_writer"
    assert writer_for("points") == "napari-svg.svg_writer"


def test_save_cursor_selection_labels_as_svg_via_viewer(
    make_napari_viewer, tmp_path
):
    """Saving a cursor-selection Labels layer to ``.svg`` goes through napari."""
    viewer = make_napari_viewer()
    layer = viewer.add_layer(_cursor_selection_labels())
    viewer.layers.selection = {layer}
    path = str(tmp_path / "selection.svg")

    written = viewer.layers.save(path, selected=True)

    assert written == [path]
    assert "<svg" in (tmp_path / "selection.svg").read_text()


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
    # Base name matching a layer name -> just "<layer>.ome.tif".
    paths = write_ome_tiff(str(tmp_path / "img1.ome.tif"), layers)
    names = {os.path.basename(p) for p in paths}
    assert names == {"img1.ome.tif", "img2.ome.tif"}


def test_write_ometif_single_layer_keeps_its_path(
    make_napari_viewer, tmp_path
):
    """A single layer is written to the path given, whatever the viewer has
    selected.

    Regression: with several layers selected in the viewer, the layer name
    was appended to a custom name the export widget had already built
    ("<custom>_<layer>.ome.tif").
    """
    viewer = make_napari_viewer()
    l1 = viewer.add_layer(_phasor_image("img1"))
    l2 = viewer.add_layer(_phasor_image("img2"))
    viewer.layers.selection = {l1, l2}
    paths = write_ome_tiff(str(tmp_path / "img1 masked.ome.tif"), l1)
    assert os.path.basename(paths[0]) == "img1 masked.ome.tif"


def test_export_csv_skips_a_fully_nan_harmonic(tmp_path):
    """A harmonic with no finite pixel contributes no rows at all."""
    import pandas as pd
    from napari.layers import Image

    from napari_phasors._writer import export_layer_as_csv

    rng = np.random.default_rng(1)
    G = rng.random((2, 4, 4))
    S = rng.random((2, 4, 4))
    # The second harmonic was never computed for this layer.
    G[1] = np.nan
    S[1] = np.nan

    layer = Image(
        np.ones((4, 4)),
        name="half_nan",
        metadata={
            "G": G,
            "S": S,
            "G_original": G.copy(),
            "S_original": S.copy(),
            "harmonics": [1, 2],
        },
    )

    out = export_layer_as_csv(str(tmp_path / "out.csv"), layer)
    df = pd.read_csv(out[0])

    assert set(df["harmonic"]) == {1}
    assert len(df) == 16


def test_write_ometif_mapping_filter_stack_roundtrip(tmp_path):
    """The metric filter stack survives an OME-TIFF roundtrip.

    The stack is what the phasor arrays are derived from, so it has to come
    back with the file: a reopened layer whose criteria were lost would show
    pixels the user had filtered out, with nothing on screen to explain it.
    """
    from napari_phasors._mapping_filters import (
        get_filters,
        new_filter,
        set_filters,
    )

    time_constants = [0.1, 1, 10]
    raw_flim_data = make_raw_flim_data(time_constants=time_constants)
    harmonic = [1, 2]
    layer = make_intensity_layer_with_phasors(raw_flim_data, harmonic=harmonic)

    stored = set_filters(
        layer,
        [
            new_filter(
                "Normal Lifetime",
                0.5,
                3.25,
                harmonic=2,
                params={'frequency': 80.0},
            ),
            new_filter("Modulation", 0.1, 0.9, mode="exclude", enabled=False),
        ],
    )

    filepath = str(tmp_path / "mapping_filters.ome.tif")
    write_ome_tiff(filepath, [(layer.data, {"metadata": layer.metadata})])

    reader = napari_get_reader(filepath, harmonics=harmonic)
    restored_layer = reader(filepath)[0]

    class _Restored:
        metadata = restored_layer[1]["metadata"]

    restored = get_filters(_Restored())
    assert restored == stored
    assert restored[0]['harmonic'] == 2
    assert restored[0]['params'] == {'frequency': 80.0}
    assert restored[1]['mode'] == "exclude"
    assert restored[1]['enabled'] is False


def _write_filter_stack_file(tmp_path):
    """Export a layer carrying a Phasor Mapping and a FRET filter."""
    from napari_phasors._mapping_filters import (
        FRET_EFFICIENCY,
        compute_metric,
        new_filter,
        set_filters,
    )

    raw_flim_data = make_raw_flim_data(time_constants=[0.1, 1, 10])
    layer = make_intensity_layer_with_phasors(raw_flim_data, harmonic=[1, 2])
    real = layer.metadata['G'][0]
    imag = layer.metadata['S'][0]
    fret_params = {'frequency': 80.0, 'donor_lifetime': 2.0}
    modulation = compute_metric("Modulation", real, imag)
    efficiency = compute_metric(
        FRET_EFFICIENCY, real, imag, params=fret_params
    )
    set_filters(
        layer,
        [
            new_filter("Modulation", 0.0, float(np.nanmedian(modulation))),
            new_filter(
                FRET_EFFICIENCY,
                0.0,
                float(np.nanmax(efficiency)),
                params=fret_params,
            ),
        ],
    )
    filepath = str(tmp_path / "filter_stack.ome.tif")
    write_ome_tiff(filepath, [(layer.data, {"metadata": layer.metadata})])
    return filepath


def _stack_mask(metadata):
    """Return the pixels the stored stack drops, measured on the originals."""
    from napari_phasors._mapping_filters import (
        combined_mask,
        filters_from_settings,
    )

    return combined_mask(
        filters_from_settings(metadata['settings']),
        metadata['original_mean'],
        metadata['G_original'],
        metadata['S_original'],
        metadata['harmonics'],
    )


def test_read_ometif_applies_mapping_filter_stack(tmp_path):
    """A stored filter stack is applied to the arrays when the file is read.

    The criteria are listed as active as soon as the file is open, so the
    pixels they drop have to be gone too, not only after a toggle.
    """
    filepath = _write_filter_stack_file(tmp_path)
    metadata = napari_get_reader(filepath)(filepath)[0][1]["metadata"]

    mask = _stack_mask(metadata)
    assert mask is not None and mask.any() and not mask.all()
    expected_nan = mask | np.isnan(metadata['original_mean'])
    assert np.array_equal(np.isnan(metadata['G'][0]), expected_nan)
    assert np.array_equal(np.isnan(metadata['S'][1]), expected_nan)
    # The originals stay unfiltered, so the stack can still be edited.
    assert not np.isnan(metadata['G_original'][0][mask]).any()


def test_read_ometif_filter_stack_shown_and_applied_in_tabs(
    tmp_path, make_napari_viewer
):
    """Both tabs show the stored criteria on, and the data is filtered."""
    from napari_phasors._mapping_filters import FRET_EFFICIENCY
    from napari_phasors.plotter import PlotterWidget

    filepath = _write_filter_stack_file(tmp_path)
    data, kwargs = napari_get_reader(filepath)(filepath)[0][:2]
    viewer = make_napari_viewer()
    plotter = PlotterWidget(viewer)
    layer = viewer.add_image(data, **kwargs)

    mask = _stack_mask(layer.metadata)
    assert np.isnan(layer.data[mask]).all()
    assert np.isnan(layer.metadata['G'][0][mask]).all()

    # These tabs refresh lazily, when they are opened.
    plotter.tab_widget.setCurrentWidget(plotter.phasor_mapping_tab)
    mapping_on = [
        f['metric']
        for f in plotter.phasor_mapping_tab.filter_list.filters()
        if f['enabled']
    ]
    plotter.tab_widget.setCurrentWidget(plotter.fret_tab)
    fret_on = [
        f['metric']
        for f in plotter.fret_tab.filter_list.filters()
        if f['enabled']
    ]
    # Opening the tabs rebuilt nothing that dropped the stack.
    assert np.isnan(layer.metadata['G'][0][mask]).all()
    assert "Modulation" in mapping_on
    assert fret_on == [FRET_EFFICIENCY]


def test_read_ometif_warns_about_a_filter_it_cannot_apply(tmp_path):
    """A lifetime filter stored without a frequency is skipped, with a warning."""
    from napari_phasors._mapping_filters import new_filter, set_filters

    raw_flim_data = make_raw_flim_data(time_constants=[0.1, 1, 10])
    layer = make_intensity_layer_with_phasors(raw_flim_data, harmonic=[1, 2])
    set_filters(layer, [new_filter("Normal Lifetime", 0.5, 3.0)])
    filepath = str(tmp_path / "unusable_filter.ome.tif")
    write_ome_tiff(filepath, [(layer.data, {"metadata": layer.metadata})])

    with pytest.warns(UserWarning, match="Normal Lifetime"):
        metadata = napari_get_reader(filepath)(filepath)[0][1]["metadata"]

    # The criterion is kept, and no pixel was dropped by it.
    assert _stack_mask(metadata) is None
    assert not np.isnan(metadata['G'][0]).all()


# ---------------------------------------------------------------------------
# Labels exported as images keep their label values
# ---------------------------------------------------------------------------


def _mask_labels(dtype=np.int32, name="mask"):
    from napari.layers import Labels

    data = np.zeros((12, 20), dtype=dtype)
    data[2:6, 3:9] = 1
    data[7:11, 10:19] = 3
    return Labels(data, name=name)


def test_export_labels_as_tiff_keeps_label_values(tmp_path):
    import tifffile

    from napari_phasors._writer import export_layer_as_image

    labels = _mask_labels()
    out = export_layer_as_image(str(tmp_path / "mask.tif"), labels)

    saved = tifffile.imread(out[0])
    assert saved.shape == labels.data.shape
    assert saved.dtype == labels.data.dtype
    np.testing.assert_array_equal(saved, labels.data)


def test_export_labels_as_png_is_16_bit_label_values(tmp_path):
    from PIL import Image

    from napari_phasors._writer import export_layer_as_image

    labels = _mask_labels()
    out = export_layer_as_image(str(tmp_path / "mask.png"), labels)

    with Image.open(out[0]) as picture:
        saved = np.array(picture)
    assert saved.dtype == np.uint16
    np.testing.assert_array_equal(saved, labels.data)


def test_export_labels_beyond_16_bit_need_tiff(tmp_path):
    """PNG cannot hold labels above 65535; TIFF keeps them."""
    import tifffile
    from napari.layers import Labels

    from napari_phasors._writer import export_layer_as_image

    labels = Labels(np.array([[0, 70000], [1, 2]], dtype=np.int32))
    with pytest.raises(ValueError, match="TIFF"):
        export_layer_as_image(str(tmp_path / "big.png"), labels)

    out = export_layer_as_image(str(tmp_path / "big.tif"), labels)
    np.testing.assert_array_equal(tifffile.imread(out[0]), labels.data)


def test_export_labels_jpeg_stays_a_picture(tmp_path):
    from PIL import Image

    from napari_phasors._writer import export_layer_as_image

    out = export_layer_as_image(str(tmp_path / "mask.jpg"), _mask_labels())
    with Image.open(out[0]) as picture:
        assert picture.mode == "RGB"


def test_export_labels_layer_data_tuple_as_tiff(tmp_path):
    import tifffile

    from napari_phasors._writer import export_layer_as_image

    labels = _mask_labels()
    out = export_layer_as_image(
        str(tmp_path / "mask.tif"),
        (labels.data, {"name": "mask"}, "labels"),
    )
    np.testing.assert_array_equal(tifffile.imread(out[0]), labels.data)


def test_export_labels_multidimensional_exports_current_slice(tmp_path):
    import tifffile
    from napari.layers import Labels

    from napari_phasors._writer import export_layer_as_image

    data = np.zeros((3, 6, 7), dtype=np.uint16)
    data[1, 2:4, 2:5] = 5
    out = export_layer_as_image(
        str(tmp_path / "stack.tif"), Labels(data), current_step=(1, 0, 0)
    )
    np.testing.assert_array_equal(tifffile.imread(out[0]), data[1])


@pytest.mark.filterwarnings("ignore:projection mode")
def test_exported_labels_image_converts_back_to_labels(tmp_path):
    """Open the exported image and convert it: the Labels layer is restored."""
    import tifffile
    from napari.components import LayerList
    from napari.layers import Image
    from napari.layers._layer_actions import _convert_to_labels

    from napari_phasors._writer import export_layer_as_image

    labels = _mask_labels()
    out = export_layer_as_image(str(tmp_path / "mask.tif"), labels)

    layers = LayerList()
    image = Image(tifffile.imread(out[0]), name="mask")
    layers.append(image)
    layers.selection = {image}
    _convert_to_labels(layers)

    assert len(layers) == 1
    np.testing.assert_array_equal(layers[0].data, labels.data)
