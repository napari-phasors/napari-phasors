"""Tests for reading Shapes and Labels layers back from SVG files."""

import numpy as np
import pytest
from napari.layers import Labels, Shapes
from napari_svg import napari_write_shapes

from napari_phasors._svg_reader import (
    napari_get_svg_reader,
    svg_file_reader,
)
from napari_phasors._writer import export_layer_as_image


def _labels(dtype=np.int32, **kwargs):
    data = np.zeros((12, 20), dtype=dtype)
    data[2:6, 3:9] = 1
    data[7:11, 10:19] = 3
    return Labels(data, name="mask", **kwargs)


def _write_shapes_svg(path, layer):
    """Write a Shapes layer the way napari-svg does."""
    meta = {
        key: getattr(layer, key)
        for key in (
            "face_color",
            "edge_color",
            "edge_width",
            "opacity",
            "shape_type",
            "z_index",
            "scale",
            "translate",
        )
    }
    meta["rotate"] = layer.rotate
    meta["shear"] = layer.shear
    meta["affine"] = layer.affine.affine_matrix
    napari_write_shapes(str(path), layer.data, meta)


# -- reader registration ----------------------------------------------------


def test_get_svg_reader_only_claims_svg_files():
    assert napari_get_svg_reader("mask.svg") is svg_file_reader
    assert napari_get_svg_reader("MASK.SVG") is svg_file_reader
    assert napari_get_svg_reader(["a.svg", "b.svg"]) is svg_file_reader
    assert napari_get_svg_reader("mask.png") is None
    assert napari_get_svg_reader(["a.svg", "b.png"]) is None


def test_reader_is_registered_for_svg_in_the_manifest():
    from npe2 import PluginManager

    pm = PluginManager.instance()
    pm.discover()
    readers = pm.iter_compatible_readers(["mask.svg"])
    assert "napari-phasors.get_svg_reader" in {r.command for r in readers}


# -- labels -----------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.uint8, np.int32, np.uint64])
def test_labels_svg_round_trip_is_exact(tmp_path, dtype):
    labels = _labels(
        dtype=dtype, scale=(0.5, 0.25), units=("micrometer", "micrometer")
    )
    path = tmp_path / "mask.svg"
    export_layer_as_image(str(path), labels)

    ((data, kwargs, layer_type),) = svg_file_reader(str(path))

    assert layer_type == "labels"
    assert data.dtype == labels.data.dtype
    np.testing.assert_array_equal(data, labels.data)
    assert kwargs["name"] == "mask"
    assert kwargs["scale"] == (0.5, 0.25)
    assert kwargs["units"] == ("micrometer", "micrometer")


def test_labels_svg_keeps_ids_a_colormap_would_repeat(tmp_path):
    data = np.arange(300, dtype=np.int32).reshape(15, 20)
    path = tmp_path / "many.svg"
    export_layer_as_image(str(path), Labels(data, name="many"))

    ((restored, _, _),) = svg_file_reader(str(path))

    np.testing.assert_array_equal(restored, data)


def test_labels_svg_of_a_stack_restores_the_exported_slice(tmp_path):
    data = np.zeros((3, 6, 7), dtype=np.uint16)
    data[1, 2:4, 2:5] = 5
    path = tmp_path / "stack.svg"
    export_layer_as_image(
        str(path),
        Labels(data, name="stack", scale=(1, 2, 3)),
        current_step=(1, 0, 0),
    )

    ((restored, kwargs, _),) = svg_file_reader(str(path))

    np.testing.assert_array_equal(restored, data[1])
    assert kwargs["scale"] == (2.0, 3.0)


def test_labels_svg_name_with_special_characters(tmp_path):
    labels = _labels()
    labels.name = 'a "quoted" <name> & more'
    path = tmp_path / "special.svg"
    export_layer_as_image(str(path), labels)

    ((_, kwargs, _),) = svg_file_reader(str(path))

    assert kwargs["name"] == labels.name


# -- shapes -----------------------------------------------------------------


def _shapes():
    layer = Shapes(
        [
            np.array([[5, 5], [5, 20], [15, 20]]),
            np.array([[20, 30], [30, 45]]),
        ],
        shape_type=["polygon", "rectangle"],
        name="shapes",
        scale=(2, 3),
        translate=(1, 4),
    )
    layer.add_ellipses(np.array([[10, 10], [20, 10], [20, 30], [10, 30]]))
    layer.add_lines(np.array([[1, 1], [9, 9]]))
    layer.add_paths(np.array([[2, 2], [3, 9], [8, 4]]))
    layer.add_rectangles(np.array([[10, 10], [14, 18], [20, 15], [16, 7.0]]))
    return layer


def test_shapes_svg_round_trip(tmp_path):
    layer = _shapes()
    path = tmp_path / "shapes.svg"
    _write_shapes_svg(path, layer)

    ((data, kwargs, layer_type),) = svg_file_reader(str(path))
    restored = Shapes(data, **kwargs)

    assert layer_type == "shapes"
    assert list(restored.shape_type) == list(layer.shape_type)
    assert kwargs["name"] == "shapes"
    np.testing.assert_allclose(restored.scale, layer.scale)
    np.testing.assert_allclose(restored.translate, layer.translate)
    np.testing.assert_array_equal(
        restored.to_labels((40, 60)), layer.to_labels((40, 60))
    )


def test_shapes_svg_with_two_layers_gives_one_layer_each(tmp_path):
    from napari_svg.hook_implementations import writer

    one = Shapes([np.array([[2, 2], [2, 9], [8, 9]])], shape_type="polygon")
    two = Shapes([np.array([[10, 10], [20, 30]])], shape_type="rectangle")
    path = tmp_path / "two.svg"
    layer_data = [layer.as_layer_data_tuple() for layer in (one, two)]
    writer(str(path), layer_data)

    layers = svg_file_reader(str(path))

    assert [kwargs["name"] for _, kwargs, _ in layers] == ["two 1", "two 2"]
    assert all(layer_type == "shapes" for _, _, layer_type in layers)


def test_shapes_svg_rotation_is_warned_about(tmp_path):
    """A rotation a Shapes layer cannot hold is warned about; scale is kept."""
    path = tmp_path / "rotated.svg"
    path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg">'
        '<g transform="rotate(30) scale(2 2)">'
        '<polygon points="1,1 5,1 5,4"/></g></svg>'
    )

    with pytest.warns(UserWarning, match="rotation or skew"):
        ((_, kwargs, layer_type),) = svg_file_reader(str(path))

    assert layer_type == "shapes"
    assert kwargs["scale"] == (2.0, 2.0)


def test_svg_without_shapes_or_labels_raises(tmp_path):
    path = tmp_path / "empty.svg"
    path.write_text('<svg xmlns="http://www.w3.org/2000/svg"><g/></svg>')

    with pytest.raises(ValueError, match="No shapes or labels"):
        svg_file_reader(str(path))


# -- through napari ---------------------------------------------------------


def test_open_exported_labels_svg_in_viewer(make_napari_viewer, tmp_path):
    viewer = make_napari_viewer()
    labels = _labels(scale=(2.0, 2.0))
    path = tmp_path / "mask.svg"
    export_layer_as_image(str(path), labels)

    opened = viewer.open(str(path), plugin="napari-phasors")

    assert len(opened) == 1
    assert isinstance(opened[0], Labels)
    np.testing.assert_array_equal(opened[0].data, labels.data)
