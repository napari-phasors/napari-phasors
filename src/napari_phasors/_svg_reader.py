"""
Read Shapes and Labels layers back from the SVG files napari writes.

Two kinds of file are understood:

* ``.svg`` files written by napari-svg for Shapes layers: each group of
  ``polygon``, ``rect``, ``ellipse``, ``line`` and ``polyline`` elements
  becomes a Shapes layer (shape types, scale and translate are restored).
* ``.svg`` files written by this plugin for Labels layers. They carry the exact
  label values next to the drawn picture, so the Labels layer, its scale and
  its units come back unchanged.

"""

from __future__ import annotations

import base64
import io
import os
import re
import warnings
import zlib
from xml.etree import ElementTree

import numpy as np

from ._writer import LABELS_SVG_METADATA_ID

_SHAPE_TAGS = ("polygon", "rect", "ellipse", "line", "polyline")
_TRANSFORM_PATTERN = re.compile(r"(\w+)\s*\(([^)]*)\)")
_NUMBER_SPLIT = re.compile(r"[,\s]+")
_IDENTITY_MATRIX = [1, 0, 0, 1, 0, 0]


def napari_get_svg_reader(path):
    """Return the SVG reader if ``path`` is an SVG file (or a list of them)."""
    paths = path if isinstance(path, list) else [path]
    if paths and all(str(p).lower().endswith(".svg") for p in paths):
        return svg_file_reader
    return None


def svg_file_reader(path):
    """Read the Shapes and Labels layers stored in SVG files.

    Parameters
    ----------
    path : str or list of str
        Path (or paths) of the SVG file(s).

    Returns
    -------
    list of tuple
        napari ``LayerData`` tuples ``(data, kwargs, layer_type)``.
    """
    paths = path if isinstance(path, list) else [path]
    layers = []
    for one_path in paths:
        found = _read_one_svg(str(one_path))
        if not found:
            raise ValueError(
                f"No shapes or labels found in '{os.path.basename(one_path)}'."
            )
        layers.extend(found)
    return layers


def _local(tag):
    """Return an XML tag without its namespace."""
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def _numbers(text):
    return [float(v) for v in _NUMBER_SPLIT.split(text.strip()) if v]


def _number(element, name):
    return float(element.get(name, 0))


def _transform(text):
    """Return an SVG ``transform`` as ``{function: [numbers]}``."""
    return {
        name: _numbers(args)
        for name, args in _TRANSFORM_PATTERN.findall(text or "")
    }


def _rotate_about(points_xy, rotation):
    """Rotate ``(N, 2)`` x/y points by an SVG ``rotate(angle cx cy)``."""
    if not rotation or abs(rotation[0]) < 1e-12:
        return points_xy
    angle = np.radians(rotation[0])
    cos, sin = np.cos(angle), np.sin(angle)
    matrix = np.array([[cos, -sin], [sin, cos]])
    center = np.array(rotation[1:3])
    return (points_xy - center) @ matrix.T + center


def _corners(x0, y0, x1, y1):
    """Return the four x/y corners in the order napari stores them."""
    return np.array([[x0, y0], [x0, y1], [x1, y1], [x1, y0]])


def _fold_right_angle(half_sizes, center, rotation):
    """Fold a quarter-turn rotation about the shape's own center away.

    napari-svg writes every rectangle and ellipse with a ``rotate`` (a vertical
    ellipse as a 90 degree one). Turning that into half sizes keeps the shape
    axis-aligned, as napari would draw it, rather than rotating its vertices.
    """
    if not rotation:
        return half_sizes, None
    quarters = rotation[0] / 90
    if (
        not np.allclose(rotation[1:3], center, atol=1e-3)
        or abs(quarters - round(quarters)) > 1e-4
    ):
        return half_sizes, rotation
    return (half_sizes[::-1] if round(quarters) % 2 else half_sizes), None


def _shape_from_element(element):
    """Return ``(shape_type, vertices_yx)`` of an SVG shape, or ``None``."""
    tag = _local(element.tag)
    rotation = _transform(element.get("transform")).get("rotate")

    if tag in ("polygon", "polyline"):
        points = np.array(_numbers(element.get("points"))).reshape(-1, 2)
        shape_type = "polygon" if tag == "polygon" else "path"
    elif tag == "line":
        points = np.array(
            [
                [_number(element, "x1"), _number(element, "y1")],
                [_number(element, "x2"), _number(element, "y2")],
            ]
        )
        shape_type = "line"
    elif tag == "rect":
        size = np.array(
            [_number(element, "width"), _number(element, "height")]
        )
        center = np.array([_number(element, "x"), _number(element, "y")])
        center = center + size / 2
        half, rotation = _fold_right_angle(size / 2, center, rotation)
        points = _corners(*(center - half), *(center + half))
        shape_type = "rectangle"
    elif tag == "ellipse":
        center = np.array([_number(element, "cx"), _number(element, "cy")])
        half = np.array([_number(element, "rx"), _number(element, "ry")])
        half, rotation = _fold_right_angle(half, center, rotation)
        points = _corners(*(center - half), *(center + half))
        shape_type = "ellipse"
    else:
        return None

    points = _rotate_about(points, rotation)
    # napari-svg writes the angle in single precision: drop that noise so a
    # vertex that sat on a pixel edge stays on it.
    return shape_type, np.round(points[:, ::-1], 4)


def _shapes_layer(group, name):
    """Build a Shapes ``LayerData`` tuple from an SVG group."""
    shapes = [s for s in map(_shape_from_element, group) if s is not None]
    transform = _transform(group.get("transform"))
    if any(transform.get(k, [0])[0] for k in ("rotate", "skewY")) or (
        not np.allclose(
            transform.get("matrix", _IDENTITY_MATRIX), _IDENTITY_MATRIX
        )
    ):
        warnings.warn(
            f"'{name}' was saved with a rotation or skew that is not "
            "restored; only its scale and translation are.",
            stacklevel=2,
        )
    kwargs = {
        "name": name,
        "shape_type": [shape_type for shape_type, _ in shapes],
        "scale": tuple(transform.get("scale", [1, 1])[::-1]),
        "translate": tuple(transform.get("translate", [0, 0])[::-1]),
    }
    return [vertices for _, vertices in shapes], kwargs, "shapes"


def _exact_labels_layer(metadata):
    """Rebuild the Labels layer stored by this plugin's SVG export."""
    raw = zlib.decompress(base64.b64decode(metadata.text))
    data = np.load(io.BytesIO(raw), allow_pickle=False)
    kwargs = {"name": metadata.get("data-name")}
    if metadata.get("data-scale"):
        kwargs["scale"] = tuple(map(float, metadata.get("data-scale").split()))
    if metadata.get("data-units"):
        kwargs["units"] = tuple(metadata.get("data-units").split())
    return data, kwargs, "labels"


def _read_one_svg(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    layers, groups = [], []
    for element in ElementTree.parse(path).getroot().iter():
        tag = _local(element.tag)
        if tag == "metadata" and element.get("id") == LABELS_SVG_METADATA_ID:
            layers.append(_exact_labels_layer(element))
        elif tag == "g" and any(_local(c.tag) in _SHAPE_TAGS for c in element):
            groups.append(element)
    for index, group in enumerate(groups):
        name = stem if len(groups) == 1 else f"{stem} {index + 1}"
        layers.append(_shapes_layer(group, name))
    return layers
