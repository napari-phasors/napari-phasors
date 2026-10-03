"""
Read Shapes and Labels layers back from the SVG files napari writes.

Two kinds of file are understood:

* ``.svg`` files written by napari-svg for Shapes layers: each group of
  ``polygon``, ``rect``, ``ellipse``, ``line`` and ``polyline`` elements
  becomes a Shapes layer (shape types, colors, edge width, opacity, scale and
  translate are restored).
* ``.svg`` files written by this plugin for Labels layers. They carry the exact
  label values next to the drawn picture, so the Labels layer, its scale and
  its units come back unchanged. An SVG without those values (an older export
  or another tool's) only holds a colored picture, resampled to the size it was
  drawn at, so its embedded picture is opened as an RGB Image layer instead.

"""

from __future__ import annotations

import base64
import binascii
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
_RGB_PATTERN = re.compile(
    r"rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*(?:,\s*([\d.]+)\s*)?\)"
)


def napari_get_svg_reader(path):
    """Return the SVG reader if ``path`` is an SVG file (or a list of them)."""
    paths = path if isinstance(path, list) else [path]
    if paths and all(str(p).lower().endswith(".svg") for p in paths):
        return svg_file_reader
    return None


def svg_file_reader(path):
    """Read the Shapes, Labels and Image layers stored in SVG files.

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
    """Return an XML tag or attribute name without its namespace."""
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def _attribute(element, name, default=None):
    """Return an attribute whatever its namespace (``xlink:href`` or not)."""
    for key, value in element.attrib.items():
        if _local(key) == name:
            return value
    return default


def _numbers(text):
    return [float(v) for v in _NUMBER_SPLIT.split(text.strip()) if v]


def _float_attribute(element, name, default=0.0):
    value = element.get(name)
    if value is None:
        return default
    try:
        return float(value.strip().removesuffix("px"))
    except ValueError:
        return default


def _parse_transform(text):
    """Return ``(scale_xy, translate_xy, rotate_deg, is_plain)`` of a transform.

    ``is_plain`` is ``False`` when the transform holds a rotation, a skew or a
    non-identity matrix, which a Shapes layer's scale/translate cannot hold.
    """
    scale = np.ones(2)
    translate = np.zeros(2)
    rotation = None
    is_plain = True
    for name, args in _TRANSFORM_PATTERN.findall(text or ""):
        try:
            values = _numbers(args)
        except ValueError:
            is_plain = False
            continue
        if name == "scale" and values:
            scale = scale * np.array(
                [values[0], values[1] if len(values) > 1 else values[0]]
            )
        elif name == "translate" and values:
            translate = translate + np.array(
                [values[0], values[1] if len(values) > 1 else 0.0]
            )
        elif name == "rotate" and values:
            center = values[1:3] if len(values) >= 3 else [0.0, 0.0]
            rotation = (values[0], np.array(center))
            if abs(values[0]) > 1e-9:
                is_plain = False
        elif name == "skewY" and values:
            if abs(values[0]) > 1e-9:
                is_plain = False
        elif name == "matrix" and len(values) == 6:
            if not np.allclose(values, [1, 0, 0, 1, 0, 0]):
                is_plain = False
        else:
            is_plain = False
    return scale, translate, rotation, is_plain


def _rotate_about(points_xy, rotation):
    """Rotate ``(N, 2)`` x/y points by an SVG ``rotate(angle cx cy)``."""
    if rotation is None or abs(rotation[0]) < 1e-12:
        return points_xy
    angle = np.radians(rotation[0])
    cos, sin = np.cos(angle), np.sin(angle)
    matrix = np.array([[cos, -sin], [sin, cos]])
    center = rotation[1]
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
    if rotation is None:
        return half_sizes, None
    angle, pivot = rotation
    quarters = angle / 90
    if (
        not np.allclose(pivot, center, atol=1e-3)
        or abs(quarters - round(quarters)) > 1e-4
    ):
        return half_sizes, rotation
    return (half_sizes[::-1] if round(quarters) % 2 else half_sizes), None


def _points_attribute(element):
    values = _numbers(element.get("points", ""))
    if len(values) < 4 or len(values) % 2:
        return None
    return np.array(values).reshape(-1, 2)


def _shape_from_element(element):
    """Return ``(shape_type, vertices_yx)`` of an SVG shape, or ``None``."""
    tag = _local(element.tag)
    rotation = _parse_transform(element.get("transform"))[2]

    if tag == "polygon":
        points = _points_attribute(element)
        if points is None or len(points) < 3:
            return None
        shape_type = "polygon"
    elif tag == "polyline":
        points = _points_attribute(element)
        if points is None:
            return None
        shape_type = "path"
    elif tag == "line":
        points = np.array(
            [
                [
                    _float_attribute(element, "x1"),
                    _float_attribute(element, "y1"),
                ],
                [
                    _float_attribute(element, "x2"),
                    _float_attribute(element, "y2"),
                ],
            ]
        )
        shape_type = "line"
    elif tag == "rect":
        width = _float_attribute(element, "width")
        height = _float_attribute(element, "height")
        center = np.array(
            [
                _float_attribute(element, "x") + width / 2,
                _float_attribute(element, "y") + height / 2,
            ]
        )
        half, rotation = _fold_right_angle(
            np.array([width, height]) / 2, center, rotation
        )
        points = _corners(*(center - half), *(center + half))
        shape_type = "rectangle"
    elif tag == "ellipse":
        center = np.array(
            [_float_attribute(element, "cx"), _float_attribute(element, "cy")]
        )
        half, rotation = _fold_right_angle(
            np.array(
                [
                    _float_attribute(element, "rx"),
                    _float_attribute(element, "ry"),
                ]
            ),
            center,
            rotation,
        )
        points = _corners(*(center - half), *(center + half))
        shape_type = "ellipse"
    else:
        return None

    points = _rotate_about(points, rotation)
    # napari-svg writes the angle in single precision: drop that noise so a
    # vertex that sat on a pixel edge stays on it.
    return shape_type, np.round(points[:, ::-1], 4)


def _parse_color(value, default):
    """Return an RGBA array for an SVG paint (``rgb()``, hex, name, none)."""
    if value is None:
        return default
    value = value.strip()
    if value == "none":
        return np.zeros(4)
    match = _RGB_PATTERN.fullmatch(value)
    if match:
        red, green, blue, alpha = match.groups()
        return np.array(
            [
                float(red) / 255,
                float(green) / 255,
                float(blue) / 255,
                1.0 if alpha is None else float(alpha),
            ]
        )
    try:
        from matplotlib.colors import to_rgba

        return np.array(to_rgba(value))
    except ValueError:
        return default


def _shapes_layer(group, name):
    """Build a Shapes ``LayerData`` tuple from an SVG group, or ``None``."""
    shape_types, vertices = [], []
    face_colors, edge_colors, edge_widths = [], [], []
    opacity = 1.0
    for element in group:
        shape = _shape_from_element(element)
        if shape is None:
            continue
        shape_types.append(shape[0])
        vertices.append(shape[1])
        face_colors.append(_parse_color(element.get("fill"), np.zeros(4)))
        edge_colors.append(
            _parse_color(element.get("stroke"), np.array([0, 0, 0, 1.0]))
        )
        edge_widths.append(_float_attribute(element, "stroke-width", 1.0))
        opacity = _float_attribute(element, "opacity", opacity)
    if not vertices:
        return None

    scale, translate, _, is_plain = _parse_transform(group.get("transform"))
    if not is_plain:
        warnings.warn(
            f"'{name}' was saved with a rotation or skew that is not "
            "restored; only its scale and translation are.",
            stacklevel=2,
        )
    kwargs = {
        "name": name,
        "shape_type": shape_types,
        "face_color": np.array(face_colors),
        "edge_color": np.array(edge_colors),
        "edge_width": edge_widths,
        "opacity": opacity,
        "scale": tuple(scale[::-1]),
        "translate": tuple(translate[::-1]),
    }
    return vertices, kwargs, "shapes"


def _exact_labels_layer(metadata, fallback_name):
    """Rebuild the Labels layer stored by this plugin's SVG export."""
    try:
        dtype = np.dtype(metadata.get("data-dtype"))
        shape = tuple(int(n) for n in metadata.get("data-shape").split())
        raw = zlib.decompress(base64.b64decode((metadata.text or "").strip()))
        data = np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
    except (TypeError, ValueError, zlib.error, binascii.Error):
        return None

    kwargs = {"name": metadata.get("data-name") or fallback_name}
    for key, attribute in (("scale", "data-scale"), ("units", "data-units")):
        text = metadata.get(attribute)
        if not text:
            continue
        values = text.split()
        if len(values) != data.ndim:
            continue
        kwargs[key] = (
            tuple(float(v) for v in values)
            if key == "scale"
            else tuple(values)
        )
    return data, kwargs, "labels"


def _decode_picture(element):
    """Return the RGB(A) or gray array of an ``<image>`` element, or ``None``."""
    href = _attribute(element, "href", "")
    if not href.startswith("data:image"):
        return None
    try:
        from PIL import Image

        encoded = href.split(",", 1)[1]
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as picture:
            if picture.mode not in ("L", "I;16", "RGB", "RGBA"):
                picture = picture.convert("RGBA")
            return np.array(picture)
    except (IndexError, ValueError, binascii.Error, OSError):
        return None


def _picture_layer(picture, name):
    """Return an Image ``LayerData`` tuple for a picture without exact values."""
    rgb = picture.ndim == 3 and picture.shape[-1] in (3, 4)
    return picture, {"name": name, "rgb": rgb}, "image"


def _read_one_svg(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    root = ElementTree.parse(path).getroot()

    exact = None
    groups, pictures = [], []
    for element in root.iter():
        tag = _local(element.tag)
        if tag == "metadata" and element.get("id") == LABELS_SVG_METADATA_ID:
            exact = _exact_labels_layer(element, stem)
        elif tag == "g" and any(_local(c.tag) in _SHAPE_TAGS for c in element):
            groups.append(element)
        elif tag == "image":
            pictures.append(element)

    layers = []
    for index, group in enumerate(groups):
        name = stem if len(groups) == 1 else f"{stem} {index + 1}"
        layer = _shapes_layer(group, name)
        if layer is not None:
            layers.append(layer)

    if exact is not None:
        layers.append(exact)
    else:
        for index, element in enumerate(pictures):
            picture = _decode_picture(element)
            if picture is None:
                continue
            name = stem if len(pictures) == 1 else f"{stem} {index + 1}"
            layers.append(_picture_layer(picture, name))
    return layers
