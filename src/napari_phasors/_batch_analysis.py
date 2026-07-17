"""Batch Analysis widget.

This module provides :class:`BatchAnalysisWidget`, a dockable widget that
applies the same phasor analysis pipeline to every supported file found in a
folder and exports the results to disk.

The widget runs *headlessly*: it reuses the plugin's reader
(:func:`napari_phasors._reader.napari_get_reader`), the separable computation
functions (:func:`napari_phasors._utils.apply_filter_and_threshold`,
:func:`napari_phasors._utils.compute_calibration_parameters`,
:func:`napari_phasors._utils.apply_calibration_correction`) and the writer
functions (:func:`napari_phasors._writer.write_ome_tiff`,
:func:`napari_phasors._writer.export_layer_as_csv`) without instantiating the
interactive analysis tabs.
"""

import ast
import contextlib
import csv
import os
import shutil
import tempfile
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from napari.layers import Image, Labels
from napari.utils.notifications import show_error, show_info
from phasorpy.cluster import phasor_cluster_gmm
from phasorpy.component import phasor_component_fit, phasor_component_fraction
from phasorpy.cursor import (
    mask_from_circular_cursor,
    mask_from_elliptic_cursor,
    mask_from_polar_cursor,
)
from phasorpy.lifetime import (
    phasor_from_fret_donor,
    phasor_from_lifetime,
    phasor_to_apparent_lifetime,
    phasor_to_normal_lifetime,
)
from phasorpy.phasor import (
    phasor_center,
    phasor_nearest_neighbor,
    phasor_to_polar,
)
from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QColor, QDoubleValidator
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from superqt import QToggleSwitch

from ._reader import extension_mapping, napari_get_reader
from ._utils import (
    LIFETIME_OUTPUT_TYPES,
    CheckableComboBox,
    HistogramSettingsDialog,
    HistogramWidget,
    PopoutWindowMixin,
    apply_calibration_correction,
    apply_filter_and_threshold,
    compute_calibration_parameters,
    make_solid_contour_cmap,
    normalize_rgb,
    populate_colormap_combobox,
    read_ome_tiff_settings,
    required_component_harmonics,
    resolve_colormap_by_name,
)
from ._writer import (
    export_layer_as_csv,
    export_layer_as_image,
    write_ome_tiff,
)
from .selection_tab import ColorButton

if TYPE_CHECKING:
    import napari


# napari builds some colormaps through a CIE-LAB -> sRGB conversion that clips a
# few out-of-gamut values and emits a UserWarning. The clipping is harmless for
# the exported images, so silence that specific, noisy warning (it would
# otherwise fire repeatedly during a batch run).
warnings.filterwarnings(
    "ignore",
    message="Conversion from CIE-LAB.*",
    category=UserWarning,
)


# Per-format read parameters exposed as typed inputs in the batch widget.
# Mirrors the ``args_defaults`` declared for each reader in
# ``napari_phasors._reader.extension_mapping``. Each spec is
# ``(arg_name, kind, default)`` where ``kind`` is one of
# ``"int"``, ``"bool"``, ``"int_or_none"``, ``"str"`` or ``"str_or_none"``.
BATCH_READER_PARAMS = {
    ".ptu": [("frame", "int", -1), ("keepdims", "bool", False)],
    ".fbd": [
        ("frame", "int", -1),
        ("keepdims", "bool", False),
        ("channel", "int_or_none", None),
    ],
    ".lif": [("image", "int_or_none", None), ("dim", "str", "λ")],
    ".json": [("channel", "int", 0), ("dtype", "str_or_none", None)],
    ".ifli": [("channel", "int", 0)],
}


def supported_extensions():
    """Return the sorted set of file extensions the plugin can read.

    Longer (more specific) extensions such as ``.ome.tif`` are placed before
    their shorter counterparts so that callers matching by suffix can pick the
    most specific match first.
    """
    exts = set(extension_mapping["raw"].keys()) | set(
        extension_mapping["processed"].keys()
    )
    return sorted(exts, key=len, reverse=True)


def match_extension(filename):
    """Return the supported extension matching ``filename``, or ``None``.

    Matching is done by longest suffix so that e.g. ``image.ome.tif`` resolves
    to ``.ome.tif`` rather than ``.tif``.
    """
    lower = filename.lower()
    for ext in supported_extensions():
        if lower.endswith(ext):
            return ext
    return None


def scan_folder(folder, recursive):
    """Scan ``folder`` for supported files grouped by extension.

    Parameters
    ----------
    folder : str
        Directory to scan.
    recursive : bool
        Whether to descend into subfolders.

    Returns
    -------
    dict
        Mapping of extension -> sorted list of absolute file paths.
    """
    found = {}
    if recursive:
        walker = ((root, files) for root, _dirs, files in os.walk(folder))
    else:
        try:
            entries = os.listdir(folder)
        except OSError:
            return found
        files = [f for f in entries if os.path.isfile(os.path.join(folder, f))]
        walker = [(folder, files)]

    for root, files in walker:
        for name in files:
            ext = match_extension(name)
            if ext is not None:
                found.setdefault(ext, []).append(os.path.join(root, name))
    for ext in found:
        found[ext].sort()
    return found


def parse_harmonics(text):
    """Parse a harmonics string into ``list[int]``, ``"all"`` or ``None``.

    Accepts comma-separated integers (e.g. ``"1, 2"``), the literal ``"all"``
    or an empty string (returns ``None``, letting the reader use its default).
    """
    text = text.strip().lower()
    if not text:
        return None
    if text == "all":
        return "all"
    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    return values or None


@dataclass
class BatchPipeline:
    """Resolved, headless analysis steps applied to every file in a batch.

    Steps are applied in order: calibration, filter/threshold, then component
    analysis. ``calibration`` is either ``{"phi_zero": ..., "mod_zero": ...}``
    (pre-computed correction parameters) or ``None``.
    """

    calibration: dict | None = None
    mask: dict | None = None
    filter: dict | None = None
    components: dict | None = None
    mapping: dict | None = None
    fret: dict | None = None
    selection: dict | None = None
    warnings: list = field(default_factory=list)


# Image file extensions accepted as batch masks.
MASK_EXTENSIONS = (".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp", ".npy")


# Phasor-mapping output quantities. The lifetime ones (shared via
# ``LIFETIME_OUTPUT_TYPES``) require a frequency.
MAPPING_OUTPUT_TYPES = [
    "Apparent Phase Lifetime",
    "Apparent Modulation Lifetime",
    "Normal Lifetime",
    "Phase",
    "Modulation",
]
MAPPING_LIFETIME_TYPES = LIFETIME_OUTPUT_TYPES

# Default cycle of cursor colors (hex) for new selection cursors.
DEFAULT_CURSOR_COLORS = [
    "#ff0000",
    "#00aa00",
    "#0066ff",
    "#ffaa00",
    "#aa00ff",
    "#00cccc",
    "#ff66cc",
    "#888800",
]


@contextlib.contextmanager
def _pipeline_step(step_name):
    """Annotate any error with the analysis step (tab) that raised it.

    Batch failures are otherwise reported with only the file name and a bare
    exception message, which does not say which enabled analysis (Calibration,
    Filter, Components, Phasor Mapping, FRET, Selection) actually failed. This
    re-raises with that context so the user knows where to look.
    """
    try:
        yield
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"{step_name} failed: {exc}") from exc


def apply_pipeline(layer, pipeline):
    """Apply ``pipeline`` to ``layer`` in place and return extra output layers.

    Parameters
    ----------
    layer : napari.layers.Image
        Layer with phasor metadata to process.
    pipeline : BatchPipeline
        The resolved analysis steps.

    Returns
    -------
    list of napari.layers.Image
        Additional layers produced by the analyses (e.g. component fraction
        images) that should also be exported. Empty if none.
    """
    extra_layers = []

    if pipeline.calibration is not None:
        with _pipeline_step("Calibration"):
            apply_calibration_correction(
                layer,
                pipeline.calibration["phi_zero"],
                pipeline.calibration["mod_zero"],
                calibration_harmonics=pipeline.calibration.get("harmonics"),
            )

    if pipeline.filter is not None:
        with _pipeline_step("Filter / Threshold"):
            apply_filter_and_threshold(layer, **pipeline.filter)

    if pipeline.mask is not None:
        with _pipeline_step("Image mask"):
            _apply_image_mask(
                layer, pipeline.mask["array"], pipeline.mask["invert"]
            )

    if pipeline.components is not None:
        with _pipeline_step("Components"):
            extra_layers.extend(
                _apply_component_fraction(layer, pipeline.components)
            )

    if pipeline.mapping is not None:
        with _pipeline_step("Phasor Mapping"):
            extra_layers.extend(_apply_phasor_mapping(layer, pipeline.mapping))

    if pipeline.fret is not None:
        with _pipeline_step("FRET"):
            extra_layers.extend(_apply_fret(layer, pipeline.fret))

    if pipeline.selection is not None:
        with _pipeline_step("Selection"):
            extra_layers.extend(_apply_selection(layer, pipeline.selection))

    return extra_layers


def _load_mask_array(path):
    """Read a mask image file and return a 2-D array (>0 means inside mask)."""
    if path.lower().endswith(".npy"):
        mask = np.load(path)
    else:
        import imageio.v3 as iio

        mask = np.asarray(iio.imread(path))
    # Reduce any channel/alpha axis: a pixel is inside the mask if any
    # channel is non-zero.
    while mask.ndim > 2:
        mask = mask.any(axis=-1)
    return mask


def _apply_image_mask(layer, mask, invert=False):
    """Set phasor pixels outside ``mask`` to NaN, mirroring the plotter.

    ``mask`` is a 2-D array; pixels where ``mask > 0`` are kept (or excluded
    when ``invert`` is True). Updates ``data``, ``G`` and ``S`` in place and
    records the mask in ``layer.metadata``.
    """
    mask = np.asarray(mask)
    if mask.shape != tuple(layer.data.shape):
        raise ValueError(
            f"mask shape {mask.shape} does not match image shape "
            f"{tuple(layer.data.shape)}"
        )

    mask_invalid = mask > 0 if invert else mask <= 0
    layer.metadata["mask"] = mask.copy()
    layer.metadata["mask_invert"] = bool(invert)

    layer.data = np.where(mask_invalid, np.nan, layer.data)
    # Mask the mean intensity image (no harmonic axis) if present.
    for key in ("original_mean", "mean"):
        array = layer.metadata.get(key)
        if array is not None and np.shape(array) == mask_invalid.shape:
            layer.metadata[key] = np.where(mask_invalid, np.nan, array)

    # Mask both the working and original phasor coordinates so the masked
    # region is excluded from analyses *and* persisted on export (the writer
    # stores the ``*_original`` arrays).
    for key in ("G", "S", "G_original", "S_original"):
        array = layer.metadata.get(key)
        if array is None:
            continue
        if array.ndim > layer.data.ndim:
            invalid = mask_invalid[np.newaxis, ...]
        else:
            invalid = mask_invalid
        layer.metadata[key] = np.where(invalid, np.nan, array)


def _select_harmonic_arrays(layer, harmonic):
    """Return ``(real, imag)`` for ``harmonic`` from a layer's phasor data."""
    g_array = layer.metadata.get("G")
    s_array = layer.metadata.get("S")
    if g_array is None or s_array is None:
        return None, None
    harmonics = layer.metadata.get("harmonics")
    if harmonics is not None:
        harmonics_array = np.atleast_1d(harmonics)
        idx = np.where(harmonics_array == harmonic)[0]
        if g_array.ndim == layer.data.ndim + 1 and idx.size > 0:
            return g_array[idx[0]], s_array[idx[0]]
    return g_array, s_array


def _apply_component_fraction(layer, components):
    """Run an N-component analysis on ``layer`` and return fraction layers.

    Two analysis types are supported, mirroring the interactive Components tab:

    - ``"linear"`` (exactly 2 components): a linear-projection fraction via
      :func:`phasorpy.component.phasor_component_fraction`, producing a single
      fraction image for the first component.
    - ``"fit"`` (N components): a multi-component fit via
      :func:`phasorpy.component.phasor_component_fit`, producing one fraction
      image per component. Fitting more than 3 components needs more than one
      harmonic: ``components["harmonics"]`` then lists the harmonics and
      ``component_real``/``component_imag`` are 2-D ``(n_harmonics,
      n_components)`` arrays of the component locations at each harmonic.

    Returns a list of :class:`napari.layers.Image` (empty if the layer lacks
    usable phasor data).
    """
    names = components["names"]
    component_real = components["component_real"]
    component_imag = components["component_imag"]
    colormaps = components.get("colormaps") or []
    contrast = components.get("contrast_limits")

    def _cmap(index):
        return colormaps[index] if index < len(colormaps) else None

    # Resolve the per-pixel phasor coordinates. A multi-harmonic fit stacks the
    # selected harmonics along a leading axis to match the 2-D component arrays.
    harmonics_list = components.get("harmonics")
    if harmonics_list and len(harmonics_list) > 1:
        reals, imags = [], []
        for harmonic in harmonics_list:
            real_h, imag_h = _select_harmonic_arrays(layer, harmonic)
            if real_h is None:
                return []
            reals.append(np.asarray(real_h))
            imags.append(np.asarray(imag_h))
        real = np.stack(reals, axis=0)
        imag = np.stack(imags, axis=0)
    else:
        harmonic = components["harmonic"]
        real, imag = _select_harmonic_arrays(layer, harmonic)
        if real is None:
            return []

    if components["analysis_type"] == "linear":
        fraction = phasor_component_fraction(
            real, imag, component_real, component_imag
        )
        # Linear projection returns the first component's fraction; the second
        # component's fraction is its complement. Export both so each gets its
        # own image, statistics and histogram.
        outputs = [
            _make_output_image(
                fraction,
                f"{names[0]} fraction: {layer.name}",
                colormap=_cmap(0),
                contrast_limits=contrast,
            )
        ]
        if len(names) > 1:
            # The second component's colormap is the inverse of the first so
            # the two fraction images share the gradient of the component line.
            outputs.append(
                _make_output_image(
                    1.0 - np.asarray(fraction),
                    f"{names[1]} fraction: {layer.name}",
                    colormap=_reversed_colormap(_cmap(0)),
                    contrast_limits=contrast,
                )
            )
        return outputs

    mean = layer.metadata.get("original_mean")
    if mean is None:
        return []
    fractions = phasor_component_fit(
        mean, real, imag, component_real, component_imag
    )
    layers = []
    for index, (fraction, name) in enumerate(
        zip(fractions, names, strict=False)
    ):
        layers.append(
            _make_output_image(
                fraction,
                f"{name} fraction: {layer.name}",
                colormap=_cmap(index),
                contrast_limits=contrast,
            )
        )
    return layers


def _reversed_colormap(name):
    """Return a reversed napari ``Colormap`` for ``name`` (or ``name`` itself).

    Used for the second Linear Projection component so its image colormap is
    the inverse of the first component's, matching the gradient drawn along the
    line between the two components.
    """
    if not name:
        return name
    try:
        from napari.utils import Colormap
        from napari.utils.colormaps import ensure_colormap

        base = ensure_colormap(name)
        colors = np.asarray(base.colors)[::-1].copy()
        return Colormap(colors, name=f"{name}_r")
    except Exception:  # noqa: BLE001 - fall back to the original name
        return name


def _make_output_image(data, name, colormap=None, contrast_limits=None):
    """Build an Image layer with optional colormap and contrast limits."""
    kwargs = {"name": name}
    if colormap:
        kwargs["colormap"] = colormap
    if contrast_limits is not None and contrast_limits[0] < contrast_limits[1]:
        kwargs["contrast_limits"] = tuple(contrast_limits)
    return Image(np.asarray(data), **kwargs)


def _compute_mapping_values(real, imag, output_type, frequency, harmonic):
    """Return the mapped value array for one output type."""
    with np.errstate(divide="ignore", invalid="ignore"):
        if output_type in ("Phase", "Modulation"):
            phase, modulation = phasor_to_polar(real, imag)
            return phase if output_type == "Phase" else modulation
        effective_frequency = frequency * harmonic
        if output_type == "Normal Lifetime":
            values = phasor_to_normal_lifetime(
                real, imag, frequency=effective_frequency
            )
        else:
            phase_lifetime, modulation_lifetime = phasor_to_apparent_lifetime(
                real, imag, frequency=effective_frequency
            )
            values = (
                phase_lifetime
                if output_type == "Apparent Phase Lifetime"
                else modulation_lifetime
            )
        return np.clip(values, a_min=0, a_max=None)


def _mapping_output_types(mapping):
    """Return the list of selected mapping output types."""
    types = mapping.get("output_types")
    if types:
        return types
    single = mapping.get("output_type")
    return [single] if single else []


def _apply_phasor_mapping(layer, mapping):
    """Map phasor coordinates to one image per selected output type.

    Mirrors ``PhasorMappingWidget.calculate_output_data`` headlessly. Returns
    a list of :class:`napari.layers.Image` (one per output type), empty if the
    layer lacks usable phasor data.
    """
    harmonic = mapping["harmonic"]
    real, imag = _select_harmonic_arrays(layer, harmonic)
    if real is None:
        return []

    layers = []
    for output_type in _mapping_output_types(mapping):
        values = _compute_mapping_values(
            real, imag, output_type, mapping["frequency"], harmonic
        )
        layers.append(
            _make_output_image(
                values,
                f"{output_type}: {layer.name}",
                colormap=mapping.get("colormap"),
                contrast_limits=mapping.get("contrast_limits"),
            )
        )
    return layers


def _apply_fret(layer, fret):
    """Compute a FRET-efficiency image for ``layer``.

    Mirrors :meth:`fret_tab.FretWidget.calculate_fret_efficiency` headlessly:
    builds the donor trajectory with
    :func:`phasorpy.lifetime.phasor_from_fret_donor` and assigns each pixel the
    efficiency of its nearest trajectory point via
    :func:`phasorpy.phasor.phasor_nearest_neighbor`.
    """
    harmonic = fret["harmonic"]
    real, imag = _select_harmonic_arrays(layer, harmonic)
    if real is None:
        return []

    efficiencies = np.linspace(0, 1, 500)
    effective_frequency = fret["frequency"] * harmonic
    neighbor_real, neighbor_imag = phasor_from_fret_donor(
        effective_frequency,
        fret["donor_lifetime"],
        fret_efficiency=efficiencies,
        donor_background=fret["donor_background"],
        background_real=fret["background_real"],
        background_imag=fret["background_imag"],
        donor_fretting=fret["donor_fretting"],
    )
    fret_efficiency = phasor_nearest_neighbor(
        real, imag, neighbor_real, neighbor_imag, values=efficiencies
    )
    name = f"FRET efficiency: {layer.name}"
    return [
        _make_output_image(
            fret_efficiency,
            name,
            colormap=fret.get("colormap"),
            contrast_limits=fret.get("contrast_limits"),
        )
    ]


def _apply_selection(layer, selection):
    """Build a cursor-selection labels image for ``layer``.

    Mirrors ``SelectionWidget`` headlessly. In ``"manual"`` mode each cursor
    (circular/elliptic/polar) labels its region with an integer id; in
    ``"cluster"`` mode GMM clusters (``phasor_cluster_gmm``) are turned
    into elliptic-cursor regions. Returns a single
    :class:`napari.layers.Labels` (empty list if no usable phasor data).
    """
    harmonic = selection["harmonic"]
    real, imag = _select_harmonic_arrays(layer, harmonic)
    if real is None:
        return []

    selection_map = np.zeros(np.shape(real), dtype=np.uint32)

    if selection.get("mode") == "cluster":
        cluster = selection.get("cluster", {})
        real_flat = np.asarray(real).ravel()
        imag_flat = np.asarray(imag).ravel()
        finite = np.isfinite(real_flat) & np.isfinite(imag_flat)
        center_real, center_imag, radius, radius_minor, angle = (
            phasor_cluster_gmm(
                real_flat[finite],
                imag_flat[finite],
                clusters=cluster.get("clusters", 2),
                sigma=cluster.get("sigma", 2.0),
            )
        )
        color_dict = {None: (0.0, 0.0, 0.0, 0.0)}
        for idx in range(len(center_real)):
            mask = mask_from_elliptic_cursor(
                real,
                imag,
                [center_real[idx]],
                [center_imag[idx]],
                radius=[radius[idx]],
                radius_minor=[radius_minor[idx]],
                angle=[angle[idx]],
            )[0]
            selection_map[mask] = idx + 1
            color_dict[idx + 1] = _cursor_rgba(
                None, idx, cluster.get("colors")
            )
        name = f"Cluster selection: {layer.name}"
        return [_make_selection_labels(selection_map, name, color_dict)]

    cursors = selection["cursors"]
    color_dict = {None: (0.0, 0.0, 0.0, 0.0)}
    for idx, cursor in enumerate(cursors):
        selection_map[_cursor_mask(real, imag, cursor)] = idx + 1
        color_dict[idx + 1] = _cursor_rgba(cursor.get("color"), idx)

    name = f"Cursor selection: {layer.name}"
    return [_make_selection_labels(selection_map, name, color_dict)]


def _cursor_rgba(color, index, palette=None):
    """Return an RGBA tuple for a cursor/cluster color.

    ``color`` is a hex/named color string (or ``None`` to fall back to
    ``palette`` or the default cursor palette by ``index``).
    """
    from matplotlib.colors import to_rgba

    if color is None:
        palette = palette or DEFAULT_CURSOR_COLORS
        color = palette[index % len(palette)]
    return tuple(float(c) for c in to_rgba(color))


def _make_selection_labels(selection_map, name, color_dict):
    """Build a Labels layer whose ids map to the cursor/cluster colors.

    Mirrors the interactive Selection tab, which colors the selection mask with
    a :class:`napari.utils.DirectLabelColormap`, so the exported image uses the
    same colors as the cursors drawn on the phasor plot.
    """
    from napari.utils import DirectLabelColormap

    return Labels(
        selection_map,
        name=name,
        colormap=DirectLabelColormap(
            color_dict=color_dict, name="selection_colors"
        ),
    )


def _cursor_mask(real, imag, cursor):
    """Return the boolean mask of pixels inside ``cursor`` (one cursor)."""
    cursor_type = cursor.get("type", "circular")
    if cursor_type == "elliptic":
        return mask_from_elliptic_cursor(
            real,
            imag,
            [cursor["g"]],
            [cursor["s"]],
            radius=[cursor["radius"]],
            radius_minor=[cursor["radius_minor"]],
            angle=[cursor["angle"]],
        )[0]
    if cursor_type == "polar":
        return mask_from_polar_cursor(
            real,
            imag,
            [cursor["phase_min"]],
            [cursor["phase_max"]],
            [cursor["modulation_min"]],
            [cursor["modulation_max"]],
        )[0]
    return mask_from_circular_cursor(
        real,
        imag,
        [cursor["g"]],
        [cursor["s"]],
        radius=[cursor["radius"]],
    )[0]


def _selection_statistics(layer, selection, selection_map):
    """Return per-region selection statistics rows for one layer.

    Mirrors the interactive Selection tab's tables: for each manual cursor the
    pixel count inside its region and that count as a percentage of the layer's
    valid (finite) pixels; for GMM clustering the same per cluster, computed
    from the produced ``selection_map`` (with each cluster's centroid). Manual
    cursor counts are computed from each cursor's own mask (so they match the
    interactive per-cursor table even where cursors overlap).
    """
    harmonic = selection["harmonic"]
    real, imag = _select_harmonic_arrays(layer, harmonic)
    if real is None:
        return []
    valid = np.isfinite(real) & np.isfinite(imag)
    total = int(np.count_nonzero(valid))

    def _percent(count):
        return (count / total * 100.0) if total else 0.0

    rows = []
    if selection.get("mode") == "cluster":
        selection_map = np.asarray(selection_map)
        ids = [int(i) for i in np.unique(selection_map) if i != 0]
        for region_id in ids:
            region = selection_map == region_id
            count = int(np.count_nonzero(region))
            rows.append(
                {
                    "region": f"Cluster {region_id}",
                    "type": "cluster",
                    "g": float(np.nanmean(real[region])) if count else "",
                    "s": float(np.nanmean(imag[region])) if count else "",
                    "radius": "",
                    "count": count,
                    "percent": _percent(count),
                }
            )
        return rows

    for idx, cursor in enumerate(selection.get("cursors", []), start=1):
        mask = _cursor_mask(real, imag, cursor)
        count = int(np.count_nonzero(mask & valid))
        rows.append(
            {
                "region": f"Cursor {idx}",
                "type": cursor.get("type", "circular"),
                "g": cursor.get("g", ""),
                "s": cursor.get("s", ""),
                "radius": cursor.get("radius", ""),
                "count": count,
                "percent": _percent(count),
            }
        )
    return rows


class BatchReadOptionsWidget(QWidget):
    """Collect reader options for a single file format (no signal preview).

    Renders typed inputs for the chosen extension's known parameters (from
    :data:`BATCH_READER_PARAMS`) plus a dynamic set of "additional kwargs"
    rows for anything else, mirroring the dynamic-kwargs feature of the
    interactive Custom Import widget.
    """

    def __init__(self, parent=None):
        """Build the empty options form; call ``set_extension`` to fill it."""
        super().__init__(parent)
        self._extension = None
        self._param_widgets = {}
        self._kwargs_rows = []

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)

        self._form_container = QWidget()
        self._form = QFormLayout(self._form_container)
        self._form.setContentsMargins(0, 0, 0, 0)
        self._form.setLabelAlignment(Qt.AlignLeft)
        self._form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        self._layout.addWidget(self._form_container)

        # Dynamic "additional kwargs" rows.
        self._kwargs_container = QWidget()
        self._kwargs_layout = QVBoxLayout(self._kwargs_container)
        self._kwargs_layout.setContentsMargins(0, 0, 0, 0)
        self._layout.addWidget(self._kwargs_container)

        kwarg_row = QHBoxLayout()
        add_kwarg_button = QPushButton("+ Add reader keyword argument")
        add_kwarg_button.clicked.connect(lambda: self._add_kwarg_row())
        kwarg_row.addWidget(add_kwarg_button)
        kwarg_row.addStretch()
        self._layout.addLayout(kwarg_row)

    def set_extension(self, extension):
        """Rebuild the typed inputs for ``extension``."""
        self._extension = extension
        # Clear existing typed inputs.
        while self._form.rowCount():
            self._form.removeRow(0)
        self._param_widgets = {}

        for arg, kind, default in BATCH_READER_PARAMS.get(extension, []):
            widget = self._make_param_widget(kind, default)
            self._param_widgets[arg] = (widget, kind)
            self._form.addRow(QLabel(f"{arg}:"), widget)

    def _make_param_widget(self, kind, default):
        """Return an input widget of type *kind* pre-set to *default*.

        Anything other than "int" or "bool" falls back to a free-text line
        edit, which ``get_reader_options`` parses back to the right type.
        """
        if kind == "int":
            w = QSpinBox()
            w.setRange(-1, 1_000_000)
            w.setValue(int(default))
            return w
        if kind == "bool":
            w = QCheckBox()
            w.setChecked(bool(default))
            return w
        # int_or_none, str, str_or_none -> free text line edit.
        w = QLineEdit()
        if default is not None:
            w.setText(str(default))
        return w

    def _add_kwarg_row(self, key="", value=""):
        """Append an additional reader kwarg row, pre-filled with *key*/*value*."""
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        key_edit = QLineEdit()
        key_edit.setPlaceholderText("key")
        key_edit.setText(key)
        val_edit = QLineEdit()
        val_edit.setPlaceholderText("value")
        val_edit.setText(value)
        remove = QPushButton("✕")
        remove.setFixedWidth(28)
        layout.addWidget(key_edit)
        layout.addWidget(val_edit)
        layout.addWidget(remove)
        self._kwargs_layout.addWidget(row)
        entry = (row, key_edit, val_edit)
        self._kwargs_rows.append(entry)
        remove.clicked.connect(lambda: self._remove_kwarg_row(entry))

    def _remove_kwarg_row(self, entry):
        """Delete *entry*'s additional reader kwarg row."""
        row, _, _ = entry
        if entry in self._kwargs_rows:
            self._kwargs_rows.remove(entry)
        row.setParent(None)
        row.deleteLater()

    def get_reader_options(self):
        """Return the collected reader options as a dict."""
        options = {}
        for arg, (widget, kind) in self._param_widgets.items():
            if kind == "int":
                options[arg] = widget.value()
            elif kind == "bool":
                options[arg] = widget.isChecked()
            else:
                text = widget.text().strip()
                if kind == "int_or_none":
                    options[arg] = int(text) if text else None
                elif kind == "str_or_none":
                    options[arg] = text if text else None
                else:  # str
                    options[arg] = text
        for _row, key_edit, val_edit in self._kwargs_rows:
            key = key_edit.text().strip()
            val_str = val_edit.text().strip()
            if not key:
                continue
            try:
                value = ast.literal_eval(val_str)
            except (ValueError, TypeError, SyntaxError, MemoryError):
                value = val_str
            options[key] = value
        return options


class CopySettingsDialog(QDialog):
    """Modal chooser for copying analysis settings from a layer or OME-TIFF."""

    def __init__(self, layer_names, parent=None):
        """Build the dialog offering *layer_names* or an OME-TIFF as source."""
        super().__init__(parent)
        self.setWindowTitle("Copy analysis settings")
        self._file_path = ""

        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                "Copy the analysis settings stored in a phasor layer or an "
                "OME-TIFF written by napari-phasors. The values populate the "
                "analysis tabs, where you can review and adjust them."
            )
        )

        layer_row = QHBoxLayout()
        layer_row.addWidget(QLabel("From layer:"))
        self.layer_combo = QComboBox()
        self.layer_combo.addItem("— Select layer —", None)
        for name in layer_names:
            self.layer_combo.addItem(name, name)
        self.layer_combo.currentIndexChanged.connect(self._on_layer_changed)
        layer_row.addWidget(self.layer_combo, 1)
        layout.addLayout(layer_row)

        file_row = QHBoxLayout()
        browse = QPushButton("From OME-TIFF…")
        browse.clicked.connect(self._browse)
        self.file_label = QLabel("<i>No file selected</i>")
        self.file_label.setWordWrap(True)
        file_row.addWidget(browse)
        file_row.addWidget(self.file_label, 1)
        layout.addLayout(file_row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _browse(self):
        """Prompt for an OME-TIFF source, clearing any layer selection."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select OME-TIFF settings source",
            "",
            "OME-TIFF Files (*.ome.tif *.ome.tiff)",
        )
        if path:
            self._file_path = path
            self.file_label.setText(path)
            self.layer_combo.blockSignals(True)
            self.layer_combo.setCurrentIndex(0)
            self.layer_combo.blockSignals(False)

    def _on_layer_changed(self, _index):
        """Clear any chosen file when a layer is picked, keeping one source."""
        if self.layer_combo.currentIndex() > 0:
            self._file_path = ""
            self.file_label.setText("<i>No file selected</i>")

    def selected_file(self):
        """Return the chosen OME-TIFF path, or empty string."""
        return self._file_path

    def selected_layer(self):
        """Return the chosen layer name, or ``None``."""
        return self.layer_combo.currentData()


def default_component_line_style():
    """Return the default component line/dot style (mirrors ComponentsWidget)."""
    return {
        "show_colormap_line": True,
        "show_component_dots": True,
        "line_offset": 0.0,
        "line_width": 3.0,
        "line_alpha": 1.0,
        "default_component_color": "dimgray",
        # Power-law gamma applied to the colormap line / fraction-histogram
        # gradient, mirroring the fraction layer's ``gamma`` in the tab.
        "colormap_gamma": 1.0,
        # Fraction histogram overlay (Linear Projection only), mirroring the
        # interactive Components tab.
        "show_fraction_histogram": False,
        "histogram_overlay_height": 0.3,
        "histogram_offset": 0.0,
        "histogram_alpha": 0.75,
    }


def default_component_label_style():
    """Return the default component label style (mirrors ComponentsWidget).

    ``show_labels`` is off by default: component names are only drawn on the
    exported phasor plot when explicitly enabled in the label-style dialog.
    """
    return {
        "show_labels": False,
        "fontsize": 10,
        "bold": False,
        "italic": False,
        "color": "black",
    }


def default_group_config():
    """Return the default grouping / display configuration."""
    return {
        "mode": "Merged",
        "assignments": {},
        "group_names": {},
        "group_colors": {},
        "layer_colors": {},
        "show_sd": True,
        "central_tendency": "None",
        "show_legend": True,
        "white_background": False,
        "smooth_curves": True,
        # Per-key contour styling (filled by the Contour Layer Settings dialog).
        "contour_layer_styles": {},  # filename -> {mode, colormap, color}
        "contour_group_styles": {},  # gid -> {mode, colormap, color}
        "contour_merged_style": "colormap",
        "contour_merged_colormap": "jet",
        "contour_merged_color": None,
    }


class _SectionLockOverlay(QWidget):
    """Transparent overlay shown over a disabled analysis section.

    It sits on top of a section's (disabled) body and intercepts clicks so the
    user gets a reminder that the analysis must be enabled before its settings
    can be edited. Each click invokes ``on_click`` (which flashes the enable
    toggle), so the reminder appears next to the toggle rather than at the
    cursor.
    """

    def __init__(self, parent, on_click):
        """Build a transparent overlay calling *on_click* when clicked."""
        super().__init__(parent)
        self._on_click = on_click
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setCursor(Qt.PointingHandCursor)

    def mousePressEvent(self, event):
        """Invoke the click callback, then defer to the base implementation."""
        if self._on_click is not None:
            self._on_click()
        super().mousePressEvent(event)


class _LockableBody(QWidget):
    """Hold an analysis section's body plus a click-catching lock overlay."""

    def __init__(self, body, on_locked_click):
        """Wrap *body* with a hidden overlay that calls *on_locked_click*."""
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(body)
        self._overlay = _SectionLockOverlay(self, on_locked_click)
        self._overlay.hide()

    def set_locked(self, locked):
        """Show (locked) or hide (unlocked) the click-catching overlay."""
        if locked:
            self._overlay.setGeometry(self.rect())
            self._overlay.raise_()
            self._overlay.show()
        else:
            self._overlay.hide()

    def resizeEvent(self, event):
        """Keep the lock overlay covering the whole body as it resizes."""
        self._overlay.setGeometry(self.rect())
        super().resizeEvent(event)


class BatchAnalysisWidget(PopoutWindowMixin, QWidget):
    """Apply a phasor analysis pipeline to every supported file in a folder."""

    # Shown as a standalone (non-dockable) window; see ``PopoutWindowMixin``.
    _popout_title = "Batch Analysis"
    _popout_max_width = 900
    _popout_min_width = 760
    _popout_height = 840

    # Display labels for the phasor-plot type comboboxes. The canonical key
    # (stored as the combobox item's userData and used throughout the export
    # pipeline) maps to a friendly label shown to the user. ``"None"`` plots
    # no data (useful to export only the phasor centers / overlays).
    PLOT_TYPE_DISPLAY = {
        "Histogram": "Density Plot (2D Histogram)",
        "Scatter": "Dot Plot (Scatter plot)",
        "Contour": "Contour Plot",
        "None": "None",
    }

    # Top-level export folder name for each analysis tab.
    _TAB_FOLDERS = {
        "components": "Component Analysis",
        "phasor_mapping": "Phasor Mapping",
        "fret": "FRET Analysis",
        "selection": "Selection Analysis",
    }

    # Per-file-type subfolder name used to group exported layer files.
    _TYPE_DIRS = {"ome.tif": "OME-TIFF", "csv": "CSV", "png": "Images"}

    # Tabs that produce a colormapped analysis image with per-tab export
    # formats (decoupled from the global Setup "Export as" formats).
    _TAB_IMAGE_TABS = {"components", "phasor_mapping", "fret", "selection"}

    def __init__(self, viewer: "napari.viewer.Viewer"):
        """Build the batch analysis window and all of its tabs."""
        super().__init__()
        self.viewer = viewer
        self._floated = False
        self._scanned = {}  # extension -> list[path]
        self._input_folder = None
        self._export_folder = None
        self._copied_calibration = None  # {"phi_zero", "mod_zero"} or None
        self._component_rows = []  # list of dicts (name/g/s widgets per row)
        self._cursor_rows = []  # list of dicts (g/s/radius widgets per row)
        self._subfolder_ref_edits = {}  # subfolder key -> QLineEdit
        self._group_config = default_group_config()
        self._mask_folders = []  # selected mask folders
        self._mask_files = []  # scanned mask file paths
        self._mask_rows = {}  # input path -> {"combo", "invert"}
        # Component overlay style (mirrors ComponentsWidget defaults)
        self._component_line_style = default_component_line_style()
        self._component_label_style = default_component_label_style()
        self._global_contrast = {}
        self._global_contrast_acc = {}
        self._auto_tabs = set()
        self._deferred_exports = []
        self._deferred_store = None
        self._deferred_dir = None
        # Signal-export state.
        self._signal_capable_cache = {}
        self._signal_export_cfg = None
        self._signal_combined = {}

        layout = QVBoxLayout(self)

        title = QLabel("<b>Batch Analysis</b>")
        title.setStyleSheet("font-size: 15px;")
        layout.addWidget(title)
        intro = QLabel(
            "Apply the same reading and analysis pipeline to every supported "
            "file in a folder, then export the results. Configure the "
            "<b>Setup</b> tab, then enable the analyses you need in their "
            "tabs. Fields marked "
            "<span style='color:#e74c3c;'>*</span> are required."
        )
        intro.setWordWrap(True)
        intro.setTextFormat(Qt.RichText)
        layout.addWidget(intro)

        self.setStyleSheet(self._cohesive_stylesheet())
        self.setMinimumWidth(360)

        self.tabs = QTabWidget()
        self.tabs.setUsesScrollButtons(True)
        self.tabs.addTab(self._build_setup_tab(), "Setup")
        self.tabs.addTab(self._build_signal_tab(), "Signal Export")
        self.tabs.addTab(
            self._build_plot_settings_tab(), "Phasor Plot Settings"
        )
        self.tabs.addTab(self._build_calibration_tab(), "Calibration")
        self.tabs.addTab(self._build_filter_tab(), "Filter && Threshold")
        self.tabs.addTab(self._build_masks_tab(), "Masks")
        self.tabs.addTab(self._build_selection_tab(), "Selection")
        self.tabs.addTab(self._build_components_tab(), "Components")
        self.tabs.addTab(self._build_mapping_tab(), "Phasor Mapping")
        self.tabs.addTab(self._build_fret_tab(), "FRET")
        layout.addWidget(self.tabs, 1)

        self._build_run_footer(layout)

        # Connect a bound method (not a lambda) so ``closeEvent`` can
        # disconnect it.

        self.viewer.layers.events.inserted.connect(self._on_layers_changed)
        self.viewer.layers.events.removed.connect(self._on_layers_changed)
        self._connect_run_enabled_signals()
        self._populate_layer_comboboxes()
        self._refresh_signal_availability()
        self._update_run_enabled()

    def _connect_run_enabled_signals(self):
        """Refresh Run-button enablement when any output option changes."""
        groups = [
            self.calibration_group,
            self.filter_group,
            self.components_group,
            self.mapping_group,
            self.fret_group,
            self.selection_group,
            self.signal_group,
        ]
        toggles = [
            self.components_plot_toggle,
            self.mapping_plot_toggle,
            self.fret_plot_toggle,
            self.selection_plot_toggle,
            self.plot_centers_checkbox,
            self.plot_individual_checkbox,
            self.plot_combined_checkbox,
            self.signal_individual_checkbox,
            self.signal_combined_checkbox,
        ]
        controls = [
            self.components_export_controls,
            self.mapping_export_controls,
            self.fret_export_controls,
        ]
        for widget in groups + toggles:
            widget.toggled.connect(lambda _=False: self._update_run_enabled())
        self.signal_format_combo.selectionChanged.connect(
            self._update_run_enabled
        )
        for control in controls:
            control["stats"].toggled.connect(
                lambda _=False: self._update_run_enabled()
            )
            for key in ("image", "histogram"):
                control[key].selectionChanged.connect(self._update_run_enabled)

    @staticmethod
    def _cohesive_stylesheet():
        """Return a theme-friendly stylesheet shared by every tab."""
        return (
            "QGroupBox {"
            "  font-weight: 600;"
            "  border: 1px solid rgba(128, 128, 128, 0.35);"
            "  border-radius: 6px;"
            "  margin-top: 10px;"
            "  padding: 8px 6px 6px 6px;"
            "}"
            "QGroupBox::title {"
            "  subcontrol-origin: margin;"
            "  subcontrol-position: top left;"
            "  left: 8px;"
            "  padding: 0 4px;"
            "}"
        )

    def _scrollable(self, widget):
        """Wrap ``widget`` in a frameless, resizable scroll area."""
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.NoFrame)
        area.setWidget(widget)
        return area

    def _enable_section(self, title, description=None, stretch_body=False):
        """Return ``(toggle, body, content)`` for an enableable analysis tab."""
        toggle = QToggleSwitch(title)
        toggle.onColor = QColor("#27ae60")
        font = toggle.font()
        font.setBold(True)
        font.setPointSizeF(font.pointSizeF() + 1)
        toggle.setFont(font)

        hint = QLabel("← Enable this analysis to edit its settings")
        hint.setStyleSheet("color: #e67e22; font-weight: 600;")
        hint.hide()

        flash_box = QWidget()
        flash_box.setAttribute(Qt.WA_StyledBackground, True)
        flash_layout = QHBoxLayout(flash_box)
        flash_layout.setContentsMargins(4, 2, 4, 2)
        flash_layout.setSpacing(8)
        flash_layout.addWidget(toggle)
        flash_layout.addWidget(hint)

        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.addWidget(flash_box)
        header_layout.addStretch()

        body = QWidget()
        body.setEnabled(False)

        flash_timer = QTimer(self)
        flash_timer.setSingleShot(True)

        def _end_flash():
            hint.hide()
            flash_box.setStyleSheet("")

        def _flash():
            hint.show()
            flash_box.setStyleSheet(
                "background: rgba(230, 126, 34, 0.30); border-radius: 5px;"
            )
            flash_timer.start(2000)

        flash_timer.timeout.connect(_end_flash)

        section = _LockableBody(body, _flash)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.addWidget(header)
        if description:
            content_layout.addWidget(self._note(description))
        content_layout.addWidget(section, 1 if stretch_body else 0)

        def _on_toggled(checked):
            body.setEnabled(checked)
            section.set_locked(not checked)
            if checked:
                flash_timer.stop()
                _end_flash()

        toggle.toggled.connect(_on_toggled)
        section.set_locked(True)
        return toggle, body, content

    @staticmethod
    def _note(text):
        """Return a wrapped, muted helper-text label (consistent across tabs)."""
        label = QLabel(text)
        label.setWordWrap(True)
        label.setStyleSheet("color: gray; font-size: 11px;")
        return label

    @staticmethod
    def _section(title):
        """Return ``(box, layout)`` for a titled section group inside a tab."""
        box = QGroupBox(title)
        layout = QVBoxLayout(box)
        return box, layout

    @staticmethod
    def _form(parent=None):
        """Return a left-aligned :class:`QFormLayout`."""
        form = QFormLayout(parent) if parent is not None else QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setLabelAlignment(Qt.AlignLeft)
        form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        return form

    def _tab_header(self, title, description):
        """Return a consistent per-tab heading with a short description."""
        container = QWidget()
        v = QVBoxLayout(container)
        v.setContentsMargins(0, 0, 0, 4)
        heading = QLabel(f"<b>{title}</b>")
        heading.setStyleSheet("font-size: 13px;")
        v.addWidget(heading)
        if description:
            v.addWidget(self._note(description))
        return container

    @staticmethod
    def _required_label(text):
        """Return a field label flagged with a required-input marker."""
        return QLabel(f"{text} <span style='color:#e74c3c;'>*</span>")

    def _phasor_export_toggle(self, overlay_name=None):
        """Return a checkbox for exporting the phasor plot of a tab."""
        if overlay_name:
            label = f"Export Phasor Plot with {overlay_name} Overlay (PNG)"
        else:
            label = "Export phasor plot (PNG)"
        toggle = QCheckBox(label)
        toggle.setChecked(True)
        return toggle

    def _make_colormap_combo(self, default="jet"):
        """Return a combobox populated with napari colormap names + icons."""
        combo = QComboBox()
        populate_colormap_combobox(
            combo, include_select_color=False, selected=default
        )
        return combo

    def _contrast_controls(self):
        """Return contrast-limit controls (auto checkbox + min/max spins).

        Returns a dict with ``auto``, ``min``, ``max`` widgets and a
        ``widget`` container laying them out on one line.
        """
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        auto = QCheckBox("Auto")
        auto.setChecked(True)
        min_spin = QDoubleSpinBox()
        min_spin.setRange(-1_000_000.0, 1_000_000.0)
        min_spin.setDecimals(4)
        min_spin.setEnabled(False)
        max_spin = QDoubleSpinBox()
        max_spin.setRange(-1_000_000.0, 1_000_000.0)
        max_spin.setDecimals(4)
        max_spin.setValue(1.0)
        max_spin.setEnabled(False)
        auto.toggled.connect(
            lambda checked: (
                min_spin.setEnabled(not checked),
                max_spin.setEnabled(not checked),
            )
        )
        layout.addWidget(auto)
        layout.addWidget(QLabel("min"))
        layout.addWidget(min_spin)
        layout.addWidget(QLabel("max"))
        layout.addWidget(max_spin)
        return {
            "auto": auto,
            "min": min_spin,
            "max": max_spin,
            "widget": container,
        }

    @staticmethod
    def _contrast_value(controls):
        """Return ``(min, max)`` from contrast controls, else ``None``."""
        if controls["auto"].isChecked():
            return None
        return (controls["min"].value(), controls["max"].value())

    def _format_combo(self, placeholder, png=True, csv=False):
        """Return a small checkable combobox offering PNG / CSV formats.

        The line edit lists the chosen formats verbatim (e.g. "PNG, CSV") and
        shows "None" when nothing is selected.
        """
        combo = CheckableComboBox(
            enable_primary_layer=False,
            placeholder=placeholder,
            unit="formats",
            no_selection_text="None",
            show_checked_list=True,
        )
        combo.addItem("PNG", checked=png)
        combo.addItem("CSV", checked=csv)
        combo.setMinimumWidth(130)
        return combo

    def _output_controls(
        self, overlay_name, image_export_label="Export image as:"
    ):
        """Return the unified per-tab Outputs group box."""
        container = QGroupBox("Outputs")
        form = self._form(container)

        image = self._format_combo("No image export", png=True, csv=False)
        image.setToolTip(
            "Formats for the colormapped analysis image (one per file). PNG is "
            "the rendered image; CSV holds the raw per-pixel values."
        )
        form.addRow(image_export_label, image)

        histogram = self._format_combo(
            "No histogram export", png=False, csv=False
        )
        histogram.setToolTip(
            "Formats for the per-file histogram of the analysis values."
        )

        groups_button = QPushButton("Configure groups and display…")
        groups_button.setToolTip(
            "Set the merged/individual/grouped display mode (global), group "
            "files for combined grouped histograms / statistics, and set "
            "histogram display options."
        )
        groups_button.clicked.connect(self._open_group_dialog)
        groups_button.setEnabled(False)
        histogram.selectionChanged.connect(
            lambda: groups_button.setEnabled(bool(histogram.checkedItems()))
        )

        hist_row = QWidget()
        hist_layout = QHBoxLayout(hist_row)
        hist_layout.setContentsMargins(0, 0, 0, 0)
        hist_layout.addWidget(histogram)
        hist_layout.addWidget(groups_button)
        hist_layout.addStretch()

        form.addRow("Export histogram as:", hist_row)

        stats = QCheckBox("Export statistics table (CSV)")
        form.addRow(stats)

        plot = self._phasor_export_toggle(overlay_name)
        form.addRow(plot)

        return {
            "widget": container,
            "image": image,
            "histogram": histogram,
            "stats": stats,
            "plot": plot,
        }

    def _build_setup_tab(self):
        """Build and return the Setup tab: input folder, format and export."""
        tab = QWidget()
        outer = QVBoxLayout(tab)
        outer.addWidget(
            self._tab_header(
                "Setup",
                "Choose the input folder and file format to process, how to "
                "read the files, and where and how to export the results. An "
                "input folder and an export folder are required to run.",
            )
        )

        copy_row = QHBoxLayout()
        self.copy_settings_button = QPushButton(
            "Copy settings from layer / OME-TIFF…"
        )
        self.copy_settings_button.setToolTip(
            "Populate the analysis tabs from the settings stored in a phasor "
            "layer or an OME-TIFF written by napari-phasors."
        )
        self.copy_settings_button.clicked.connect(self._on_copy_settings)
        copy_row.addWidget(self.copy_settings_button)
        copy_row.addStretch()
        outer.addLayout(copy_row)

        input_group = QGroupBox("Input folder")
        input_layout = QVBoxLayout(input_group)
        input_layout.addWidget(
            self._required_label("Folder containing the files to process:")
        )

        row = QHBoxLayout()
        self.select_folder_button = QPushButton("Select input folder")
        self.select_folder_button.clicked.connect(self._on_select_folder)
        row.addWidget(self.select_folder_button)
        self.folder_label = QLabel("<i>No folder selected</i>")
        self.folder_label.setWordWrap(True)
        row.addWidget(self.folder_label, 1)
        input_layout.addLayout(row)

        self.subfolders_checkbox = QCheckBox("Include subfolders")
        self.subfolders_checkbox.stateChanged.connect(lambda _: self._rescan())
        input_layout.addWidget(self.subfolders_checkbox)

        format_row = QHBoxLayout()
        format_row.addWidget(QLabel("File format:"))
        self.format_combobox = QComboBox()
        self.format_combobox.currentIndexChanged.connect(
            self._on_format_changed
        )
        format_row.addWidget(self.format_combobox)
        format_row.addStretch()
        input_layout.addLayout(format_row)
        outer.addWidget(input_group)

        # Read parameters ---------------------------------------------------
        read_group = QGroupBox("Read parameters")
        read_layout = QVBoxLayout(read_group)

        harmonics_row = QHBoxLayout()
        harmonics_row.addWidget(QLabel("Harmonics:"))
        self.harmonics_edit = QLineEdit("1")
        self.harmonics_edit.setToolTip(
            "Comma-separated harmonics (e.g. '1, 2'), 'all', or empty for the "
            "reader default."
        )
        harmonics_row.addWidget(self.harmonics_edit, 1)
        read_layout.addLayout(harmonics_row)

        self.read_options_widget = BatchReadOptionsWidget()
        read_layout.addWidget(self.read_options_widget)
        outer.addWidget(read_group)

        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout(export_group)
        export_layout.addWidget(
            self._required_label("Destination folder for the results:")
        )

        out_row = QHBoxLayout()
        self.export_folder_button = QPushButton("Select export folder")
        self.export_folder_button.clicked.connect(self._on_select_export)
        out_row.addWidget(self.export_folder_button)
        self.export_folder_label = QLabel("<i>No folder selected</i>")
        self.export_folder_label.setWordWrap(True)
        out_row.addWidget(self.export_folder_label, 1)
        export_layout.addLayout(out_row)

        types_row = QHBoxLayout()
        types_row.addWidget(
            QLabel("Export Intensty Image (with phasor data) as:")
        )
        self.export_ometiff_checkbox = QCheckBox("OME-TIFF")
        self.export_ometiff_checkbox.setChecked(True)
        self.export_csv_checkbox = QCheckBox("CSV")
        self.export_image_checkbox = QCheckBox("Image (PNG)")
        for checkbox in (
            self.export_ometiff_checkbox,
            self.export_csv_checkbox,
            self.export_image_checkbox,
        ):
            checkbox.toggled.connect(
                lambda _=False: self._update_run_enabled()
            )
            types_row.addWidget(checkbox)
        types_row.addStretch()
        export_layout.addLayout(types_row)

        self.preserve_paths_checkbox = QCheckBox(
            "Preserve relative subfolder structure"
        )
        self.preserve_paths_checkbox.setChecked(True)
        export_layout.addWidget(self.preserve_paths_checkbox)

        self.export_colorbar_checkbox = QCheckBox(
            "Include colorbar in exported analysis images (PNG)"
        )
        self.export_colorbar_checkbox.setChecked(True)
        self.export_colorbar_checkbox.setToolTip(
            "Draw a colorbar next to the colormapped analysis images "
            "(component fractions, phasor mapping, FRET) exported as PNG."
        )
        export_layout.addWidget(self.export_colorbar_checkbox)

        dpi_row = QHBoxLayout()
        dpi_row.addWidget(QLabel("Image DPI:"))
        self.export_dpi_combo = QComboBox()
        self.export_dpi_combo.addItem("Low (100 DPI)", 100)
        self.export_dpi_combo.addItem("Mid (300 DPI)", 300)
        self.export_dpi_combo.addItem("High (600 DPI)", 600)
        self.export_dpi_combo.setCurrentIndex(1)
        self.export_dpi_combo.setToolTip(
            "Resolution (dots per inch) used when rendering exported PNG "
            "images, phasor plots, and histograms. Higher DPI gives sharper "
            "images but larger files and slower exports."
        )
        self.export_dpi_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        dpi_row.addWidget(self.export_dpi_combo)
        dpi_row.addStretch()
        export_layout.addLayout(dpi_row)

        suffix_row = QHBoxLayout()
        suffix_row.addWidget(QLabel("Filename suffix:"))
        self.suffix_edit = QLineEdit()
        self.suffix_edit.setPlaceholderText(
            "text added to the name of the exported files"
        )
        suffix_row.addWidget(self.suffix_edit, 1)
        export_layout.addLayout(suffix_row)

        self.load_into_viewer_checkbox = QCheckBox(
            "Also load results into the viewer"
        )
        export_layout.addWidget(self.load_into_viewer_checkbox)
        outer.addWidget(export_group)

        performance_group = QGroupBox("Performance")
        performance_layout = QVBoxLayout(performance_group)
        self.streaming_checkbox = QCheckBox(
            "Bounded memory (streaming aggregation)"
        )
        self.streaming_checkbox.setToolTip(
            "Accumulate fixed-bin histograms instead of holding every pixel "
            "in memory for grouped contour/histogram outputs. Median and "
            "center-of-mass become histogram approximations."
        )
        performance_layout.addWidget(self.streaming_checkbox)
        workers_row = QHBoxLayout()
        workers_row.addWidget(QLabel("Parallel workers:"))
        self.threads_spin = QSpinBox()
        self.threads_spin.setRange(1, max(1, (os.cpu_count() or 1)))
        self.threads_spin.setValue(1)
        self.threads_spin.setToolTip(
            "Read and compute files concurrently (1 = single-threaded). "
            "File writing and plot rendering stay on the main thread."
        )
        workers_row.addWidget(self.threads_spin)
        workers_row.addStretch()
        performance_layout.addLayout(workers_row)
        outer.addWidget(performance_group)

        outer.addStretch()
        return self._scrollable(tab)

    def _build_signal_tab(self):
        """Build the signal-export tab (average signal along the phasor axis)."""
        self._signal_available = False

        tab = QWidget()
        outer = QVBoxLayout(tab)
        outer.addWidget(
            self._tab_header(
                "Signal export",
                "Export the average signal along the axis the phasor is "
                "computed on (time bins for FLIM, wavelength for HSI). The "
                "signal is averaged over all pixels; for raw files the batch "
                "mask is applied first.",
            )
        )

        self.signal_group, body, content = self._enable_section(
            "Enable signal export",
            "For raw files the signal is averaged per pixel over the batch "
            "mask (Masks tab); for processed OME-TIFF files the stored signal "
            "(summed over all pixels at import) is used and the mask does not "
            "apply.",
        )
        outer.addWidget(content)

        self._signal_status = QLabel()
        self._signal_status.setWordWrap(True)
        self._signal_status.setStyleSheet("color: #e74c3c; font-weight: 600;")
        self._signal_status.hide()
        outer.addWidget(self._signal_status)

        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)

        ind_box, ind_layout = self._section("Individual signal plots")
        self.signal_individual_checkbox = QCheckBox(
            "Export individual signal plots (one per file)"
        )
        self.signal_individual_checkbox.setToolTip(
            "Export one signal plot per file, drawn like the signal preview "
            "in the Custom Import widget."
        )
        ind_layout.addWidget(self.signal_individual_checkbox)

        color_row = QHBoxLayout()
        color_row.addWidget(QLabel("Line color:"))
        self.signal_color = ColorButton(QColor("#1f77b4"))
        self.signal_color.setToolTip("Color of the individual signal line.")
        color_row.addWidget(self.signal_color)
        color_row.addStretch()
        ind_layout.addLayout(color_row)
        body_layout.addWidget(ind_box)

        comb_box, comb_layout = self._section("Combined signal plot")
        self.signal_combined_checkbox = QCheckBox(
            "Export combined signal plot (mean ± shaded SD)"
        )
        self.signal_combined_checkbox.setToolTip(
            "Overlay every file's signal as a per-group mean line with a "
            "shaded ±1 standard-deviation band."
        )
        comb_layout.addWidget(self.signal_combined_checkbox)

        groups_row = QHBoxLayout()
        signal_groups_button = QPushButton("Configure Groups…")
        signal_groups_button.setToolTip(
            "Assign files to groups for the combined plot and choose the "
            "color and legend for each group (shared with the other tabs)."
        )
        signal_groups_button.clicked.connect(self._open_plot_group_dialog)
        groups_row.addWidget(signal_groups_button)
        groups_row.addStretch()
        comb_layout.addLayout(groups_row)
        comb_layout.addWidget(
            self._note(
                "Grouping is shared with the Phasor Plot Settings tab; each "
                "group is drawn as a mean line with a shaded ±1 SD band."
            )
        )
        signal_groups_button.setEnabled(
            self.signal_combined_checkbox.isChecked()
        )
        self.signal_combined_checkbox.toggled.connect(
            signal_groups_button.setEnabled
        )
        body_layout.addWidget(comb_box)

        fmt_row = QHBoxLayout()
        fmt_row.addWidget(QLabel("Export plots as:"))
        self.signal_format_combo = self._format_combo(
            "No signal export", png=True, csv=False
        )
        self.signal_format_combo.setToolTip(
            "Formats for the signal plots (individual and combined). PNG is "
            "the rendered plot; CSV holds the underlying signal values."
        )
        fmt_row.addWidget(self.signal_format_combo)
        fmt_row.addStretch()
        body_layout.addLayout(fmt_row)

        norm_row = QHBoxLayout()
        norm_row.addWidget(QLabel("Normalization:"))
        self.signal_normalize_combo = QComboBox()
        self.signal_normalize_combo.addItem("None (average per pixel)", "none")
        self.signal_normalize_combo.addItem("Peak (max = 1)", "peak")
        self.signal_normalize_combo.addItem("Area (sum = 1)", "area")
        self.signal_normalize_combo.setToolTip(
            "Scale each signal before plotting so files with different "
            "intensities are comparable (applied to individual and combined "
            "plots)."
        )
        norm_row.addWidget(self.signal_normalize_combo)
        norm_row.addStretch()
        body_layout.addLayout(norm_row)

        chan_row = QHBoxLayout()
        chan_row.addWidget(QLabel("Channels:"))
        self.signal_channel_combo = QComboBox()
        self.signal_channel_combo.addItem("Separate plots", "separate")
        self.signal_channel_combo.addItem("Together (overlaid)", "together")
        self.signal_channel_combo.setToolTip(
            "For multichannel files, draw each channel in its own plot "
            "(Separate) or overlay all channels in one plot (Together). "
            "Applies to both the individual and combined plots; single-channel "
            "files are unaffected."
        )
        chan_row.addWidget(self.signal_channel_combo)
        chan_row.addStretch()
        body_layout.addLayout(chan_row)

        outer.addStretch()
        return self._scrollable(tab)

    def _signal_file_capable(self, path, ext):
        """Return whether ``path`` can provide a signal for export (cached)."""
        if path in self._signal_capable_cache:
            return self._signal_capable_cache[path]
        processed_exts = set(extension_mapping["processed"])
        raw_exts = set(extension_mapping["raw"])
        if ext not in processed_exts or ext in raw_exts:
            result = True
        elif ext in (".ome.tif", ".ome.tiff"):
            try:
                settings = read_ome_tiff_settings(path)
                result = settings.get("summed_signal") is not None
            except Exception:  # noqa: BLE001
                result = False
        else:
            result = False
        self._signal_capable_cache[path] = result
        return result

    def _signal_availability(self, ext):
        """Return ``(available, explanation)`` for the selected format."""
        if not ext or not self._scanned.get(ext):
            return (
                False,
                "Select an input folder and file format to export signals.",
            )
        processed_exts = set(extension_mapping["processed"])
        raw_exts = set(extension_mapping["raw"])
        if ext not in processed_exts or ext in raw_exts:
            return True, ""
        if ext in (".ome.tif", ".ome.tiff"):
            if self._signal_file_capable(self._scanned[ext][0], ext):
                return True, ""
            return (
                False,
                "These OME-TIFF files do not contain a stored signal (they "
                "were not written by napari-phasors), so the signal cannot be "
                "reconstructed. Signal export is unavailable for this format.",
            )
        return (
            False,
            "This processed format stores only phasor coordinates, not the "
            "original signal, so signal export is unavailable. Use the raw "
            "files or napari-phasors OME-TIFFs instead.",
        )

    def _refresh_signal_availability(self):
        """Enable/disable the signal tab from the selected format's capability."""
        ext = self.format_combobox.currentData()
        available, explanation = self._signal_availability(ext)
        self._signal_available = available
        if available:
            self.signal_group.setEnabled(True)
            self._signal_status.hide()
        else:
            if self.signal_group.isChecked():
                self.signal_group.setChecked(False)
            self.signal_group.setEnabled(False)
            self._signal_status.setText(explanation)
            self._signal_status.show()
        self._update_run_enabled()

    def _signal_export_requested(self):
        """Whether a usable signal export is enabled for the current format."""
        return (
            bool(self._signal_available)
            and self.signal_group.isChecked()
            and bool(self.signal_format_combo.checkedItems())
            and (
                self.signal_individual_checkbox.isChecked()
                or self.signal_combined_checkbox.isChecked()
            )
        )

    @staticmethod
    def _normalize_signal(profile, mode):
        """Return ``profile`` scaled per ``mode`` ('none'/'peak'/'area')."""
        profile = np.asarray(profile, dtype=float)
        if mode == "peak":
            peak = np.nanmax(profile) if profile.size else 0.0
            return profile / peak if peak else profile
        if mode == "area":
            area = np.nansum(profile) if profile.size else 0.0
            return profile / area if area else profile
        return profile

    @staticmethod
    def _signal_ylabel(mode):
        """Return the y-axis label matching the normalization ``mode``."""
        if mode == "peak":
            return "Signal (peak-normalized)"
        if mode == "area":
            return "Signal (area-normalized)"
        return "Mean signal per pixel"

    def _build_calibration_tab(self):
        """Build and return the Calibration tab."""
        tab = QWidget()
        outer = QVBoxLayout(tab)

        self.calibration_group, body, content = self._enable_section(
            "Enable calibration",
            "Calibrate phasor coordinates against a reference of known "
            "lifetime before the rest of the pipeline runs.",
        )
        group_layout = QVBoxLayout(body)
        group_layout.setContentsMargins(0, 0, 0, 0)

        ref_box, ref_layout = self._section("Reference")

        source_row = QHBoxLayout()
        source_row.addWidget(QLabel("Reference source:"))
        self.calib_source_combo = QComboBox()
        self.calib_source_combo.addItem("Same reference for all", "same")
        self.calib_source_combo.addItem(
            "Different reference per subfolder", "subfolder"
        )
        self.calib_source_combo.addItem(
            "Use copied phase/modulation", "copied"
        )
        self.calib_source_combo.setToolTip(
            "Where the calibration reference comes from: one reference for "
            "every file, a different reference per subfolder, or the "
            "phase/modulation correction copied from existing settings."
        )
        self.calib_source_combo.currentIndexChanged.connect(
            self._on_calib_source_changed
        )
        source_row.addWidget(self.calib_source_combo)
        source_row.addStretch()
        ref_layout.addLayout(source_row)

        # Same-reference-for-all widgets.
        self.calib_same_widget = QWidget()
        same_form = self._form(self.calib_same_widget)
        self.calib_reference_combo = QComboBox()
        self.calib_reference_combo.setToolTip(
            "Phasor layer (already loaded in the viewer) to use as the "
            "calibration reference for every file."
        )
        same_form.addRow(
            self._required_label("Reference layer:"),
            self.calib_reference_combo,
        )
        calib_file_row = QHBoxLayout()
        self.calib_file_edit = QLineEdit()
        self.calib_file_edit.setPlaceholderText("…or a reference file")
        self.calib_file_edit.setToolTip(
            "Path to a reference file to read as the calibration reference "
            "instead of a loaded layer."
        )
        calib_browse = QPushButton("Browse")
        calib_browse.clicked.connect(self._on_browse_calibration_file)
        calib_file_row.addWidget(self.calib_file_edit, 1)
        calib_file_row.addWidget(calib_browse)
        calib_file_container = QWidget()
        calib_file_container.setLayout(calib_file_row)
        same_form.addRow("Reference file:", calib_file_container)
        ref_layout.addWidget(self.calib_same_widget)

        # Per-subfolder widgets.
        self.calib_subfolder_widget = QWidget()
        subfolder_layout = QVBoxLayout(self.calib_subfolder_widget)
        subfolder_layout.setContentsMargins(0, 0, 0, 0)
        subfolder_layout.addWidget(
            self._note(
                "Pick a reference file for each subfolder discovered in the "
                "input folder."
            )
        )
        self.calib_subfolder_container = QWidget()
        self.calib_subfolder_rows_layout = QVBoxLayout(
            self.calib_subfolder_container
        )
        self.calib_subfolder_rows_layout.setContentsMargins(0, 0, 0, 0)
        subfolder_layout.addWidget(self.calib_subfolder_container)
        ref_layout.addWidget(self.calib_subfolder_widget)

        # Copied-calibration note.
        self.calib_copied_label = QLabel(
            "Using calibration (phase/modulation) copied from settings. It "
            "is applied to every file."
        )
        self.calib_copied_label.setWordWrap(True)
        ref_layout.addWidget(self.calib_copied_label)
        group_layout.addWidget(ref_box)

        # Shared frequency / reference lifetime.
        lifetime_box, lifetime_layout = self._section("Reference lifetime")
        self._calib_freq_lifetime_widget = QWidget()
        fl_form = self._form(self._calib_freq_lifetime_widget)
        self.calib_frequency_spin = QLineEdit()
        self.calib_frequency_spin.setValidator(QDoubleValidator())
        self.calib_frequency_spin.setToolTip(
            "Laser repetition / modulation frequency (MHz) at which the data "
            "and reference were acquired."
        )
        fl_form.addRow(
            self._required_label("Frequency (MHz):"), self.calib_frequency_spin
        )
        self.calib_lifetime_spin = QDoubleSpinBox()
        self.calib_lifetime_spin.setRange(0.0, 1_000.0)
        self.calib_lifetime_spin.setDecimals(3)
        self.calib_lifetime_spin.setToolTip(
            "Known fluorescence lifetime (ns) of the calibration reference."
        )
        fl_form.addRow(
            self._required_label("Reference lifetime (ns):"),
            self.calib_lifetime_spin,
        )
        lifetime_layout.addWidget(self._calib_freq_lifetime_widget)
        group_layout.addWidget(lifetime_box)

        outer.addWidget(content)
        outer.addStretch()

        self._on_calib_source_changed()
        return self._scrollable(tab)

    def _build_filter_tab(self):
        """Build and return the Filter tab: filter method and thresholding."""
        tab = QWidget()
        outer = QVBoxLayout(tab)

        self.filter_group, body, content = self._enable_section(
            "Enable filter & threshold",
            "Smooth the phasor coordinates and exclude low-intensity "
            "pixels before any analysis runs.",
        )
        v = QVBoxLayout(body)
        v.setContentsMargins(0, 0, 0, 0)

        filter_box, filter_layout = self._section("Filter")
        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Filter method:"))
        self.filter_method_combo = QComboBox()
        self.filter_method_combo.addItems(
            ["None", "Median", "Wavelet (binlet pawFLIM)"]
        )
        self.filter_method_combo.setToolTip(
            "Smoothing applied to the phasor coordinates before analysis: "
            "none, a median filter, or a wavelet (binlet pawFLIM) filter."
        )
        self.filter_method_combo.currentTextChanged.connect(
            self._on_filter_method_changed
        )
        method_row.addWidget(self.filter_method_combo)
        method_row.addStretch()
        filter_layout.addLayout(method_row)

        self.median_filter_widget = QWidget()
        median_form = self._form(self.median_filter_widget)
        self.median_size_spin = QSpinBox()
        self.median_size_spin.setRange(2, 99)
        self.median_size_spin.setValue(3)
        self.median_size_spin.setToolTip(
            "Side length (in pixels) of the square median-filter window. "
            "Larger windows smooth more."
        )
        median_form.addRow("Median kernel size:", self.median_size_spin)
        self.median_repeat_spin = QSpinBox()
        self.median_repeat_spin.setRange(1, 100)
        self.median_repeat_spin.setValue(1)
        self.median_repeat_spin.setToolTip(
            "How many times the median filter is applied in succession."
        )
        median_form.addRow("Median repetitions:", self.median_repeat_spin)
        filter_layout.addWidget(self.median_filter_widget)

        self.wavelet_filter_widget = QWidget()
        wavelet_form = self._form(self.wavelet_filter_widget)
        self.wavelet_sigma_spin = QDoubleSpinBox()
        self.wavelet_sigma_spin.setRange(0.1, 10.0)
        self.wavelet_sigma_spin.setSingleStep(0.1)
        self.wavelet_sigma_spin.setValue(2.0)
        self.wavelet_sigma_spin.setToolTip(
            "Noise standard deviation used by the wavelet filter; higher "
            "values remove more noise."
        )
        wavelet_form.addRow("Wavelet sigma:", self.wavelet_sigma_spin)
        self.wavelet_levels_spin = QSpinBox()
        self.wavelet_levels_spin.setRange(1, 10)
        self.wavelet_levels_spin.setValue(1)
        self.wavelet_levels_spin.setToolTip(
            "Number of wavelet decomposition levels used for denoising."
        )
        wavelet_form.addRow("Wavelet levels:", self.wavelet_levels_spin)
        filter_layout.addWidget(self.wavelet_filter_widget)
        v.addWidget(filter_box)

        threshold_box, threshold_layout = self._section("Threshold")
        threshold_row = QHBoxLayout()
        threshold_row.addWidget(QLabel("Threshold method:"))
        self.threshold_method_combo = QComboBox()
        self.threshold_method_combo.addItems(
            ["None", "Manual", "Otsu", "Li", "Yen"]
        )
        self.threshold_method_combo.setToolTip(
            "How the intensity threshold is chosen to exclude low-signal "
            "pixels: a manual value or an automatic method (Otsu / Li / Yen)."
        )
        self.threshold_method_combo.currentTextChanged.connect(
            self._on_threshold_method_changed
        )
        threshold_row.addWidget(self.threshold_method_combo)
        threshold_row.addStretch()
        threshold_layout.addLayout(threshold_row)

        self.threshold_manual_widget = QWidget()
        threshold_form = self._form(self.threshold_manual_widget)
        self.threshold_min_spin = QDoubleSpinBox()
        self.threshold_min_spin.setRange(0.0, 1_000_000.0)
        self.threshold_min_spin.setDecimals(3)
        self.threshold_min_spin.setToolTip(
            "Minimum mean intensity a pixel must have to be kept; dimmer "
            "pixels are excluded (set to NaN)."
        )
        threshold_form.addRow("Threshold (min):", self.threshold_min_spin)
        self.threshold_max_spin = QDoubleSpinBox()
        self.threshold_max_spin.setRange(0.0, 1_000_000.0)
        self.threshold_max_spin.setDecimals(3)
        self.threshold_max_spin.setSpecialValueText("none")
        self.threshold_max_spin.setToolTip(
            "Maximum mean intensity a pixel may have to be kept; set to "
            "'none' (the minimum) to disable the upper bound."
        )
        threshold_form.addRow("Threshold (max):", self.threshold_max_spin)
        threshold_layout.addWidget(self.threshold_manual_widget)
        v.addWidget(threshold_box)

        outer.addWidget(content)
        outer.addStretch()
        self._on_filter_method_changed(self.filter_method_combo.currentText())
        self._on_threshold_method_changed(
            self.threshold_method_combo.currentText()
        )
        return self._scrollable(tab)

    def _on_filter_method_changed(self, method):
        """Show only the settings belonging to the chosen filter *method*."""
        self.median_filter_widget.setVisible(method == "Median")
        self.wavelet_filter_widget.setVisible(
            method == "Wavelet (binlet pawFLIM)"
        )

    def _on_threshold_method_changed(self, method):
        """Show the manual threshold input only for the "Manual" method."""
        self.threshold_manual_widget.setVisible(method == "Manual")

    def _build_masks_tab(self):
        """Build and return the Masks tab: mask folders and per-file pairing."""
        tab = QWidget()
        outer = QVBoxLayout(tab)

        self.masks_group, body, content = self._enable_section(
            "Enable masks",
            "Restrict each image to a region of interest using mask image "
            "files matched to inputs by name.",
            stretch_body=True,
        )
        group_layout = QVBoxLayout(body)
        group_layout.setContentsMargins(0, 0, 0, 0)

        folder_row = QHBoxLayout()
        add_button = QPushButton("Add mask folder…")
        add_button.clicked.connect(self._on_add_mask_folder)
        clear_button = QPushButton("Clear")
        clear_button.clicked.connect(self._on_clear_mask_folders)
        folder_row.addWidget(add_button)
        folder_row.addWidget(clear_button)
        folder_row.addStretch()
        group_layout.addLayout(folder_row)

        self.mask_folders_label = QLabel("<i>No mask folder selected</i>")
        self.mask_folders_label.setWordWrap(True)
        group_layout.addWidget(self.mask_folders_label)

        self.mask_subfolders_checkbox = QCheckBox("Include mask subfolders")
        self.mask_subfolders_checkbox.stateChanged.connect(
            lambda _: self._scan_mask_files()
        )
        group_layout.addWidget(self.mask_subfolders_checkbox)

        self._mask_rows_container = QWidget()
        self._mask_rows_layout = QVBoxLayout(self._mask_rows_container)
        self._mask_rows_layout.setContentsMargins(0, 0, 0, 0)
        rows_scroll = QScrollArea()
        rows_scroll.setWidgetResizable(True)
        rows_scroll.setFrameShape(QFrame.NoFrame)
        rows_scroll.setWidget(self._mask_rows_container)

        rows_scroll.setMinimumHeight(360)
        group_layout.addWidget(rows_scroll, 1)

        note = QLabel(
            "Masks are image files matched to each input by name "
            "(e.g. ABC.ptu ↔ ABC_mask.png). Pick a mask per image and "
            "optionally invert it; pixels outside the mask are excluded "
            "(set to NaN) before analysis."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: gray; font-size: 11px;")
        group_layout.addWidget(note)

        outer.addWidget(content, 1)
        return self._scrollable(tab)

    def _on_add_mask_folder(self):
        """Prompt for a mask folder and rescan, ignoring duplicates."""
        folder = QFileDialog.getExistingDirectory(self, "Select mask folder")
        if folder and folder not in self._mask_folders:
            self._mask_folders.append(folder)
            self._scan_mask_files()

    def _on_clear_mask_folders(self):
        """Drop every mask folder and rescan."""
        self._mask_folders = []
        self._scan_mask_files()

    def _scan_mask_files(self):
        """Rescan the mask folders and refresh the per-file pairing rows.

        Honours the "include subfolders" checkbox, and skips folders that
        cannot be listed.
        """
        recursive = self.mask_subfolders_checkbox.isChecked()
        files = []
        for folder in self._mask_folders:
            if recursive:
                for root, _dirs, names in os.walk(folder):
                    for name in names:
                        if name.lower().endswith(MASK_EXTENSIONS):
                            files.append(os.path.join(root, name))
            else:
                try:
                    entries = os.listdir(folder)
                except OSError:
                    continue
                for name in entries:
                    full = os.path.join(folder, name)
                    if os.path.isfile(full) and name.lower().endswith(
                        MASK_EXTENSIONS
                    ):
                        files.append(full)
        self._mask_files = sorted(set(files))
        if self._mask_folders:
            self.mask_folders_label.setText(
                f"{len(self._mask_folders)} folder(s), "
                f"{len(self._mask_files)} mask files"
            )
        else:
            self.mask_folders_label.setText("<i>No mask folder selected</i>")
        self._rebuild_mask_rows()

    @staticmethod
    def _rank_mask_candidates(image_path, mask_files):
        """Return mask files ranked by name similarity to ``image_path``."""
        image_stem = os.path.splitext(os.path.basename(image_path))[0]
        # Strip a trailing supported extension chunk like ``.ome``.
        image_stem = image_stem.split(".")[0].lower()
        scored = []
        for mask in mask_files:
            mask_stem = os.path.splitext(os.path.basename(mask))[0].lower()
            if image_stem == mask_stem:
                score = 0
            elif image_stem and (
                image_stem in mask_stem or mask_stem in image_stem
            ):
                score = 1 + abs(len(mask_stem) - len(image_stem))
            else:
                continue
            scored.append((score, mask))
        scored.sort(key=lambda item: (item[0], item[1]))
        return [mask for _score, mask in scored]

    def _rebuild_mask_rows(self):
        """Rebuild one mask-pairing row per input file, best match preselected."""
        layout = self._mask_rows_layout
        for i in reversed(range(layout.count())):
            widget = layout.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self._mask_rows = {}

        ext = self.format_combobox.currentData()
        files = self._scanned.get(ext, [])
        if not files:
            placeholder = QLabel(
                "Scan an input folder and select a format to list images."
            )
            placeholder.setStyleSheet("color: gray;")
            layout.addWidget(placeholder)
            return

        for path in files:
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            label = QLabel(os.path.basename(path))
            label.setMinimumWidth(140)
            combo = QComboBox()
            combo.addItem("None", None)
            candidates = self._rank_mask_candidates(path, self._mask_files)
            remaining = [m for m in self._mask_files if m not in candidates]
            for mask in candidates + remaining:
                combo.addItem(os.path.basename(mask), mask)
            if candidates:
                combo.setCurrentIndex(1)  # best-matched candidate
            invert = QCheckBox("Invert")
            row_layout.addWidget(label, 1)
            row_layout.addWidget(combo, 1)
            row_layout.addWidget(invert)
            layout.addWidget(row)
            self._mask_rows[path] = {"combo": combo, "invert": invert}

    def _mask_for(self, path):
        """Return ``(mask_path or None, invert)`` for an input ``path``."""
        row = self._mask_rows.get(path)
        if row is None:
            return (None, False)
        return (row["combo"].currentData(), row["invert"].isChecked())

    def _build_components_tab(self):
        """Build and return the Components tab."""
        tab = QWidget()
        outer = QVBoxLayout(tab)

        self.components_group, body, content = self._enable_section(
            "Enable component analysis",
            "Decompose each pixel into fractional contributions of known "
            "phasor components. Two components support a linear projection "
            "or a fit; three or more use a fit (more than three need "
            "locations at several harmonics).",
        )
        group_layout = QVBoxLayout(body)
        group_layout.setContentsMargins(0, 0, 0, 0)

        inputs_box, inputs_layout = self._section("Component locations")

        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("Analysis type:"))
        self.analysis_type_combo = QComboBox()
        self.analysis_type_combo.setToolTip(
            "How fractions are computed from the component locations: a linear "
            "projection (two components) or a multi-component fit."
        )
        type_row.addWidget(self.analysis_type_combo, 1)
        inputs_layout.addLayout(type_row)

        fh_row = QHBoxLayout()
        fh_row.addWidget(QLabel("Frequency (MHz):"))
        self.components_frequency_spin = QLineEdit()
        self.components_frequency_spin.setValidator(QDoubleValidator())
        self.components_frequency_spin.setText(str(80.0))
        self.components_frequency_spin.setToolTip(
            "Used to convert a typed lifetime into G/S coordinates."
        )
        fh_row.addWidget(self.components_frequency_spin)
        fh_row.addWidget(QLabel("Harmonic:"))
        self.components_harmonic_spin = QSpinBox()
        self.components_harmonic_spin.setRange(1, 100)
        self.components_harmonic_spin.setValue(1)
        self.components_harmonic_spin.setToolTip(
            "Harmonic whose component G/S locations you are editing. Fitting "
            "more than 3 components needs locations at several harmonics: "
            "switch this selector and enter G/S for each required harmonic."
        )

        self._component_current_harmonic = 1
        self.components_harmonic_spin.valueChanged.connect(
            self._on_component_harmonic_changed
        )
        fh_row.addWidget(self.components_harmonic_spin)
        fh_row.addStretch()
        inputs_layout.addLayout(fh_row)

        inputs_layout.addWidget(
            self._required_label("Component G/S locations:")
        )
        self._components_container = QWidget()
        self._components_layout = QVBoxLayout(self._components_container)
        self._components_layout.setContentsMargins(0, 0, 0, 0)
        inputs_layout.addWidget(self._components_container)

        add_button = QPushButton("+ Add component")
        add_button.clicked.connect(lambda: self._add_component_row())
        inputs_layout.addWidget(add_button)

        self.components_note = QLabel("")
        self.components_note.setWordWrap(True)
        self.components_note.setStyleSheet("color: gray; font-size: 11px;")
        inputs_layout.addWidget(self.components_note)
        group_layout.addWidget(inputs_box)

        style_section_box, style_section_layout = self._section(
            "Fraction images and phasor plot styling"
        )
        contrast_row = QHBoxLayout()
        contrast_row.addWidget(QLabel("Range (image && histogram):"))
        self.components_contrast = self._contrast_controls()
        self.components_contrast["widget"].setToolTip(
            "Fraction range for the colormapped fraction images and the "
            "histogram. Uncheck 'Auto' to set fixed min/max limits; 'Auto' "
            "pools a single range across every file so they are comparable."
        )
        contrast_row.addWidget(self.components_contrast["widget"])
        contrast_row.addStretch()
        style_section_layout.addLayout(contrast_row)

        overlay_row = QHBoxLayout()
        overlay_row.addWidget(QLabel("Phasor plot overlay:"))
        line_style_btn = QPushButton("Edit line layout…")
        line_style_btn.setToolTip(
            "Edit the dots and connecting line drawn between components in the "
            "exported phasor plot."
        )
        line_style_btn.clicked.connect(self._open_component_line_style_dialog)
        label_style_btn = QPushButton("Edit component name layout…")
        label_style_btn.setToolTip(
            "Edit the component name labels drawn in the exported phasor plot."
        )
        label_style_btn.clicked.connect(
            self._open_component_label_style_dialog
        )
        overlay_row.addWidget(line_style_btn)
        overlay_row.addWidget(label_style_btn)
        overlay_row.addStretch()
        style_section_layout.addLayout(overlay_row)
        group_layout.addWidget(style_section_box)

        self.components_export_controls = self._output_controls(
            "Components", "Export component fractions as:"
        )
        self.components_plot_toggle = self.components_export_controls["plot"]
        group_layout.addWidget(self.components_export_controls["widget"])

        outer.addWidget(content)
        outer.addStretch()

        # Start with the two-component default.
        self._add_component_row("Component 1")
        self._add_component_row("Component 2")
        self._update_component_controls()
        return self._scrollable(tab)

    def _build_mapping_tab(self):
        """Build and return the Phasor Mapping tab."""
        tab = QWidget()
        outer = QVBoxLayout(tab)

        self.mapping_group, body, content = self._enable_section(
            "Enable phasor mapping",
            "Map each pixel's phasor to a quantity (lifetime, phase or "
            "modulation) and export it as an image per file.",
        )
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)

        output_box = QGroupBox("Output images")
        output_form = self._form(output_box)

        self.mapping_output_combo = CheckableComboBox(
            enable_primary_layer=False
        )
        for index, output_type in enumerate(MAPPING_OUTPUT_TYPES):
            self.mapping_output_combo.addItem(output_type, checked=index == 0)
        self.mapping_output_combo.setToolTip(
            "Quantities to map each pixel's phasor to; one image is exported "
            "per file per selected quantity."
        )
        self.mapping_output_combo.selectionChanged.connect(
            self._on_mapping_output_changed
        )
        output_form.addRow(
            self._required_label("Outputs:"), self.mapping_output_combo
        )

        self.mapping_frequency_spin = QLineEdit()
        self.mapping_frequency_spin.setValidator(QDoubleValidator())
        self.mapping_frequency_spin.setText(str(80.0))
        self.mapping_frequency_spin.setToolTip(
            "Frequency (MHz) used to convert phasor coordinates to lifetimes "
            "(only needed for lifetime outputs)."
        )
        self.mapping_frequency_label = QLabel("Frequency (MHz):")
        output_form.addRow(
            self.mapping_frequency_label, self.mapping_frequency_spin
        )

        self.mapping_harmonic_spin = QSpinBox()
        self.mapping_harmonic_spin.setRange(1, 100)
        self.mapping_harmonic_spin.setValue(1)
        self.mapping_harmonic_spin.setToolTip(
            "Harmonic used for the mapping (one of the read harmonics)."
        )
        output_form.addRow("Harmonic:", self.mapping_harmonic_spin)

        self.mapping_colormap_combo = self._make_colormap_combo("jet")
        self.mapping_colormap_combo.setToolTip(
            "Colormap applied to the exported mapped images."
        )
        output_form.addRow("Colormap:", self.mapping_colormap_combo)
        self.mapping_contrast = self._contrast_controls()
        self.mapping_contrast["widget"].setToolTip(
            "Value range (e.g. lifetime) for the colormapped image and the "
            "histogram. Uncheck 'Auto' to set fixed min/max limits; 'Auto' "
            "pools a single range across every file so they are comparable."
        )
        output_form.addRow(
            "Range (image && histogram):", self.mapping_contrast["widget"]
        )

        note = QLabel(
            "Outputs one image per file for each selected quantity. Lifetime "
            "quantities require a frequency; Phase/Modulation do not."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: gray; font-size: 11px;")
        output_form.addRow(note)
        body_layout.addWidget(output_box)

        body_layout.addWidget(self._build_mapping_mesh_box())

        self.mapping_export_controls = self._output_controls(
            "Mapping", "Export phasor mapped as:"
        )
        self.mapping_plot_toggle = self.mapping_export_controls["plot"]
        body_layout.addWidget(self.mapping_export_controls["widget"])

        outer.addWidget(content)
        outer.addStretch()
        self._on_mapping_output_changed()
        return self._scrollable(tab)

    def _build_mapping_mesh_box(self):
        """Return a box with phase/modulation mesh + color-by controls."""
        box = QGroupBox("Phasor plot mesh and coloring")
        layout = QVBoxLayout(box)

        top_form = QFormLayout()
        top_form.setContentsMargins(0, 0, 0, 0)
        top_form.setLabelAlignment(Qt.AlignLeft)
        top_form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)

        self.mapping_color_by_combo = QComboBox()
        self.mapping_color_by_combo.addItems(["None", "Phase", "Modulation"])
        self.mapping_color_by_combo.setToolTip(
            "Color the exported phasor plot's points by phase or modulation."
        )
        top_form.addRow("Color phasor by:", self.mapping_color_by_combo)

        self.mapping_mesh_phase_checkbox = QCheckBox("Phase mesh")
        self.mapping_mesh_mod_checkbox = QCheckBox("Modulation mesh")
        mesh_row = QHBoxLayout()
        mesh_row.addWidget(self.mapping_mesh_phase_checkbox)
        mesh_row.addWidget(self.mapping_mesh_mod_checkbox)
        mesh_row.addStretch()
        mesh_container = QWidget()
        mesh_container.setLayout(mesh_row)
        top_form.addRow("Mesh overlay:", mesh_container)

        self.mapping_mesh_colormap_combo = self._make_colormap_combo("jet")
        top_form.addRow(
            "Mesh/color colormap:", self.mapping_mesh_colormap_combo
        )
        self.mapping_mesh_alpha_spin = QDoubleSpinBox()
        self.mapping_mesh_alpha_spin.setRange(0.05, 1.0)
        self.mapping_mesh_alpha_spin.setSingleStep(0.05)
        self.mapping_mesh_alpha_spin.setValue(0.45)
        top_form.addRow("Mesh alpha:", self.mapping_mesh_alpha_spin)

        self.mapping_mesh_clip_checkbox = QCheckBox("Clip mesh to semicircle")
        self.mapping_mesh_clip_checkbox.setChecked(False)
        self.mapping_mesh_clip_checkbox.setToolTip(
            "Only show the mesh inside the universal semicircle (semicircle "
            "plot geometry only)."
        )
        top_form.addRow("", self.mapping_mesh_clip_checkbox)

        self.mapping_range_auto_checkbox = QCheckBox(
            "Auto (range from all files)"
        )
        self.mapping_range_auto_checkbox.setChecked(True)
        self.mapping_range_auto_checkbox.setToolTip(
            "Compute a single phase/modulation range pooled across every file "
            "at export. Uncheck to set fixed ranges manually below."
        )
        self.mapping_range_auto_checkbox.toggled.connect(
            self._on_mapping_range_auto_toggled
        )
        top_form.addRow("Range:", self.mapping_range_auto_checkbox)
        layout.addLayout(top_form)

        self.mapping_phase_min_spin = QDoubleSpinBox()
        self.mapping_phase_min_spin.setRange(0.0, 2.0 * float(np.pi))
        self.mapping_phase_min_spin.setDecimals(2)
        self.mapping_phase_min_spin.setSingleStep(0.05)
        self.mapping_phase_min_spin.setValue(0.0)
        self.mapping_phase_max_spin = QDoubleSpinBox()
        self.mapping_phase_max_spin.setRange(0.0, 2.0 * float(np.pi))
        self.mapping_phase_max_spin.setDecimals(2)
        self.mapping_phase_max_spin.setSingleStep(0.05)
        self.mapping_phase_max_spin.setValue(round(np.pi / 2.0, 2))
        phase_row = QHBoxLayout()
        phase_row.addWidget(QLabel("Phase range (rad):"))
        phase_row.addWidget(self.mapping_phase_min_spin)
        phase_row.addWidget(QLabel("to"))
        phase_row.addWidget(self.mapping_phase_max_spin)
        phase_row.addStretch()
        layout.addLayout(phase_row)

        self.mapping_mod_min_spin = QDoubleSpinBox()
        self.mapping_mod_min_spin.setRange(0.0, 1.0)
        self.mapping_mod_min_spin.setDecimals(2)
        self.mapping_mod_min_spin.setSingleStep(0.05)
        self.mapping_mod_min_spin.setValue(0.0)
        self.mapping_mod_max_spin = QDoubleSpinBox()
        self.mapping_mod_max_spin.setRange(0.0, 1.0)
        self.mapping_mod_max_spin.setDecimals(2)
        self.mapping_mod_max_spin.setSingleStep(0.05)
        self.mapping_mod_max_spin.setValue(1.0)
        mod_row = QHBoxLayout()
        mod_row.addWidget(QLabel("Modulation range:"))
        mod_row.addWidget(self.mapping_mod_min_spin)
        mod_row.addWidget(QLabel("to"))
        mod_row.addWidget(self.mapping_mod_max_spin)
        mod_row.addStretch()
        layout.addLayout(mod_row)

        self._on_mapping_range_auto_toggled(True)

        note = QLabel(
            "When 'Export phasor plot' is on, one PNG is exported per "
            "selected mesh (and a base plot), coloring points by phase or "
            "modulation. The phase/modulation ranges restrict which mesh "
            "cells are shown."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(note)
        return box

    def _on_mapping_range_auto_toggled(self, checked):
        """Enable/disable the manual mesh range fields for the Auto checkbox."""
        for spin in (
            self.mapping_phase_min_spin,
            self.mapping_phase_max_spin,
            self.mapping_mod_min_spin,
            self.mapping_mod_max_spin,
        ):
            spin.setEnabled(not checked)

    def _resolve_mesh_ranges(self):
        """Return ``(phase_range, modulation_range)`` for the mapping mesh."""
        manual_phase = (
            self.mapping_phase_min_spin.value(),
            self.mapping_phase_max_spin.value(),
        )
        manual_mod = (
            self.mapping_mod_min_spin.value(),
            self.mapping_mod_max_spin.value(),
        )
        if not self.mapping_range_auto_checkbox.isChecked():
            return manual_phase, manual_mod
        coords = self._gather_all_phasor_coords(
            self.mapping_harmonic_spin.value()
        )
        if coords is None:
            return manual_phase, manual_mod
        g_flat, s_flat = coords
        with np.errstate(invalid="ignore"):
            phase, modulation = phasor_to_polar(g_flat, s_flat)
            phase = phase[np.isfinite(phase)]
            modulation = modulation[np.isfinite(modulation)]
        phase_range = (
            (float(np.nanmin(phase)), float(np.nanmax(phase)))
            if phase.size
            else manual_phase
        )
        mod_range = (
            (float(np.nanmin(modulation)), float(np.nanmax(modulation)))
            if modulation.size
            else manual_mod
        )
        return phase_range, mod_range

    def _auto_mapping_ranges(self):
        """Set the mesh phase/modulation ranges from *all* scanned files."""
        from napari.utils.notifications import show_warning

        harmonic = self.mapping_harmonic_spin.value()
        coords = self._gather_all_phasor_coords(harmonic)
        if coords is None:
            show_warning(
                "Scan an input folder with files of the selected format to "
                "compute the range from all data."
            )
            return
        g_flat, s_flat = coords
        with np.errstate(invalid="ignore"):
            phase, modulation = phasor_to_polar(g_flat, s_flat)
            phase = phase[np.isfinite(phase)]
            modulation = modulation[np.isfinite(modulation)]

        if phase.size:
            self.mapping_phase_min_spin.setValue(float(np.nanmin(phase)))
            self.mapping_phase_max_spin.setValue(float(np.nanmax(phase)))
        if modulation.size:
            self.mapping_mod_min_spin.setValue(float(np.nanmin(modulation)))
            self.mapping_mod_max_spin.setValue(float(np.nanmax(modulation)))

    def _gather_all_phasor_coords(self, harmonic):
        """Return pooled ``(G, S)`` for ``harmonic`` across every scanned file."""
        ext = self.format_combobox.currentData()
        files = self._scanned.get(ext, []) if ext else []
        if not files:
            return None

        try:
            harmonics = parse_harmonics(self.harmonics_edit.text())
        except ValueError:
            harmonics = None
        reader_options = self.read_options_widget.get_reader_options()

        coord_pipeline = BatchPipeline(
            filter=(
                self._collect_filter_kwargs()
                if self.filter_group.isChecked()
                else None
            )
        )
        try:
            calibration_map = self._resolve_calibration_map(harmonics)
        except ValueError:
            calibration_map = None
        masks_enabled = self.masks_group.isChecked()

        reals, imags = [], []
        for path in files:
            try:
                calibration = (
                    self._calibration_for(path, calibration_map)
                    if calibration_map
                    else None
                )
                mask_spec = (
                    self._mask_for(path) if masks_enabled else (None, False)
                )
                results = self._read_compute_file(
                    path,
                    ext,
                    reader_options,
                    harmonics,
                    coord_pipeline,
                    calibration,
                    mask_spec,
                )
            except Exception:  # noqa: BLE001 - skip unreadable files
                continue
            for layer, _extra in results:
                real, imag = _select_harmonic_arrays(layer, int(harmonic))
                if real is None:
                    continue
                real = np.asarray(real).ravel()
                imag = np.asarray(imag).ravel()
                finite = np.isfinite(real) & np.isfinite(imag)
                reals.append(real[finite])
                imags.append(imag[finite])

        if not reals:
            return None
        return np.concatenate(reals), np.concatenate(imags)

    def _build_fret_tab(self):
        """Build and return the FRET tab."""
        tab = QWidget()
        outer = QVBoxLayout(tab)

        self.fret_group, body, content = self._enable_section(
            "Enable FRET efficiency",
            "Estimate the apparent FRET efficiency of each pixel from a "
            "donor trajectory, and export it as an image per file.",
        )
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)

        inputs_box, inputs_layout = self._section("Donor trajectory")
        form = self._form()

        self.fret_donor_lifetime_spin = QDoubleSpinBox()
        self.fret_donor_lifetime_spin.setRange(0.0, 1_000.0)
        self.fret_donor_lifetime_spin.setDecimals(3)
        self.fret_donor_lifetime_spin.setValue(2.0)
        self.fret_donor_lifetime_spin.setToolTip(
            "Fluorescence lifetime (ns) of the donor in the absence of FRET; "
            "anchors the donor trajectory."
        )
        form.addRow(
            self._required_label("Donor lifetime (ns):"),
            self.fret_donor_lifetime_spin,
        )

        self.fret_frequency_spin = QLineEdit()
        self.fret_frequency_spin.setValidator(QDoubleValidator())
        self.fret_frequency_spin.setText(str(80.0))
        self.fret_frequency_spin.setToolTip(
            "Laser repetition / modulation frequency (MHz) at which the data "
            "were acquired."
        )
        form.addRow(
            self._required_label("Frequency (MHz):"), self.fret_frequency_spin
        )

        self.fret_harmonic_spin = QSpinBox()
        self.fret_harmonic_spin.setRange(1, 100)
        self.fret_harmonic_spin.setValue(1)
        self.fret_harmonic_spin.setToolTip(
            "Harmonic used to compute the FRET efficiency (one of the read "
            "harmonics)."
        )
        form.addRow("Harmonic:", self.fret_harmonic_spin)

        self.fret_background_spin = QDoubleSpinBox()
        self.fret_background_spin.setRange(0.0, 1.0)
        self.fret_background_spin.setSingleStep(0.05)
        self.fret_background_spin.setDecimals(3)
        self.fret_background_spin.setValue(0.1)
        self.fret_background_spin.setToolTip(
            "Fraction of the donor signal coming from background; shifts the "
            "trajectory toward the background phasor."
        )
        form.addRow("Donor background:", self.fret_background_spin)

        self.fret_fretting_spin = QDoubleSpinBox()
        self.fret_fretting_spin.setRange(0.0, 1.0)
        self.fret_fretting_spin.setSingleStep(0.05)
        self.fret_fretting_spin.setDecimals(3)
        self.fret_fretting_spin.setValue(1.0)
        self.fret_fretting_spin.setToolTip(
            "Fraction of donor molecules that undergo FRET (1.0 = all donors "
            "participate)."
        )
        form.addRow("Donor fretting fraction:", self.fret_fretting_spin)

        self.fret_bg_real_spin = QDoubleSpinBox()
        self.fret_bg_real_spin.setRange(-2.0, 2.0)
        self.fret_bg_real_spin.setDecimals(4)
        self.fret_bg_real_spin.setToolTip(
            "G (real) coordinate of the background phasor."
        )
        form.addRow("Background G:", self.fret_bg_real_spin)

        self.fret_bg_imag_spin = QDoubleSpinBox()
        self.fret_bg_imag_spin.setRange(-2.0, 2.0)
        self.fret_bg_imag_spin.setDecimals(4)
        self.fret_bg_imag_spin.setToolTip(
            "S (imaginary) coordinate of the background phasor."
        )
        form.addRow("Background S:", self.fret_bg_imag_spin)
        inputs_layout.addLayout(form)
        group_layout = body_layout
        group_layout.addWidget(inputs_box)

        style_box, style_layout = self._section("Image styling")
        style_form = self._form()
        self.fret_colormap_combo = self._make_colormap_combo("viridis")
        self.fret_colormap_combo.setToolTip(
            "Colormap applied to the exported FRET-efficiency images."
        )
        style_form.addRow("Colormap:", self.fret_colormap_combo)
        self.fret_contrast = self._contrast_controls()
        self.fret_contrast["widget"].setToolTip(
            "Efficiency range for the colormapped image and the histogram. "
            "Uncheck 'Auto' to set fixed min/max limits; 'Auto' pools a single "
            "range across every file so the outputs are comparable."
        )
        style_form.addRow(
            "Range (image && histogram):", self.fret_contrast["widget"]
        )
        style_layout.addLayout(style_form)
        group_layout.addWidget(style_box)

        self.fret_export_controls = self._output_controls(
            "FRET", "Export FRET efficiencies as:"
        )
        self.fret_plot_toggle = self.fret_export_controls["plot"]
        group_layout.addWidget(self.fret_export_controls["widget"])

        outer.addWidget(content)
        outer.addStretch()
        return self._scrollable(tab)

    def _on_mapping_output_changed(self):
        """Enable the frequency input only if an output selected needs it."""
        selected = self.mapping_output_combo.checkedItems()
        requires_frequency = any(
            output in MAPPING_LIFETIME_TYPES for output in selected
        )
        self.mapping_frequency_label.setEnabled(requires_frequency)
        self.mapping_frequency_spin.setEnabled(requires_frequency)

    def _build_selection_tab(self):
        """Build and return the Selection tab: selection mode and cursors."""
        tab = QWidget()
        outer = QVBoxLayout(tab)

        self.selection_group, body, content = self._enable_section(
            "Enable cursor selection",
            "Label phasor regions with manual cursors or automatic GMM "
            "clustering, and export a selection (labels) image per file.",
        )
        group_layout = QVBoxLayout(body)
        group_layout.setContentsMargins(0, 0, 0, 0)

        inputs_box, inputs_layout = self._section("Selection definition")

        config_row = QHBoxLayout()
        config_row.addWidget(QLabel("Harmonic:"))
        self.selection_harmonic_spin = QSpinBox()
        self.selection_harmonic_spin.setRange(1, 100)
        self.selection_harmonic_spin.setValue(1)
        self.selection_harmonic_spin.setToolTip(
            "Harmonic whose phasor coordinates the cursors / clustering act "
            "on (one of the read harmonics)."
        )
        config_row.addWidget(self.selection_harmonic_spin)
        config_row.addSpacing(12)
        config_row.addWidget(QLabel("Mode:"))
        self.selection_mode_combo = QComboBox()
        self.selection_mode_combo.addItem("Manual cursors", "manual")
        self.selection_mode_combo.addItem(
            "Automatic clustering (GMM)", "cluster"
        )
        self.selection_mode_combo.setToolTip(
            "Define regions with manually placed cursors, or detect them "
            "automatically with Gaussian-mixture (GMM) clustering."
        )
        self.selection_mode_combo.currentIndexChanged.connect(
            self._on_selection_mode_changed
        )
        config_row.addWidget(self.selection_mode_combo, 1)
        inputs_layout.addLayout(config_row)

        # Manual cursors.
        self.selection_manual_widget = QWidget()
        manual_layout = QVBoxLayout(self.selection_manual_widget)
        manual_layout.setContentsMargins(0, 0, 0, 0)
        manual_layout.addWidget(self._required_label("Cursors:"))
        self._cursors_container = QWidget()
        self._cursors_layout = QVBoxLayout(self._cursors_container)
        self._cursors_layout.setContentsMargins(0, 0, 0, 0)
        manual_layout.addWidget(self._cursors_container)
        add_button = QPushButton("+ Add cursor")
        add_button.clicked.connect(lambda: self._add_cursor_row())
        manual_layout.addWidget(add_button)
        inputs_layout.addWidget(self.selection_manual_widget)

        # Automatic clustering (GMM).
        self.selection_cluster_widget = QWidget()
        cluster_form = self._form(self.selection_cluster_widget)
        self.cluster_count_spin = QSpinBox()
        self.cluster_count_spin.setRange(2, 100)
        self.cluster_count_spin.setValue(2)
        self.cluster_count_spin.setToolTip(
            "Number of clusters (regions) the GMM fits to the phasor data."
        )
        cluster_form.addRow("Number of clusters:", self.cluster_count_spin)
        self.cluster_sigma_spin = QDoubleSpinBox()
        self.cluster_sigma_spin.setRange(0.1, 10.0)
        self.cluster_sigma_spin.setSingleStep(0.1)
        self.cluster_sigma_spin.setValue(2.0)
        self.cluster_sigma_spin.setToolTip(
            "Size of each cluster's elliptic region, in standard deviations "
            "of the fitted Gaussian."
        )
        cluster_form.addRow("Sigma:", self.cluster_sigma_spin)
        inputs_layout.addWidget(self.selection_cluster_widget)

        inputs_layout.addWidget(
            self._note(
                "Manual circular/elliptic/polar cursors, or automatic GMM "
                "clusters, label their region in a selection image (one id "
                "each). Angles are in degrees; polar phase bounds are in "
                "radians."
            )
        )
        group_layout.addWidget(inputs_box)

        outputs_box, outputs_layout = self._section("Outputs")
        form = self._form()

        self.selection_export_combo = self._format_combo(
            "No image export", png=True, csv=False
        )
        self.selection_export_combo.setToolTip(
            "Formats for the selection image (one per file). PNG is the rendered image; CSV holds the label IDs."
        )
        self.selection_export_combo.selectionChanged.connect(
            self._update_run_enabled
        )
        form.addRow(
            "Export cursor selections as:", self.selection_export_combo
        )

        self.selection_stats_checkbox = QCheckBox(
            "Export selection statistics (CSV)"
        )
        self.selection_stats_checkbox.setToolTip(
            "Write one CSV with, per cursor / cluster, the pixel count inside "
            "the region and its percentage of the valid pixels — like the "
            "table in the interactive Selection tab."
        )
        self.selection_stats_checkbox.toggled.connect(
            lambda _=False: self._update_run_enabled()
        )
        form.addRow(self.selection_stats_checkbox)
        self.selection_plot_toggle = self._phasor_export_toggle("Selection")
        form.addRow(self.selection_plot_toggle)
        outputs_layout.addLayout(form)
        group_layout.addWidget(outputs_box)

        outer.addWidget(content)
        outer.addStretch()

        self._add_cursor_row()
        self._on_selection_mode_changed()
        return self._scrollable(tab)

    def _on_selection_mode_changed(self):
        """Show only the settings belonging to the chosen selection mode."""
        mode = self.selection_mode_combo.currentData()
        self.selection_manual_widget.setVisible(mode == "manual")
        self.selection_cluster_widget.setVisible(mode == "cluster")

    def _add_cursor_row(self, g=0.0, s=0.0, radius=0.05):
        """Append a cursor row centred at (*g*, *s*) with the given *radius*."""
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        row_layout = QHBoxLayout(frame)
        row_layout.setContentsMargins(4, 2, 4, 2)

        index = len(self._cursor_rows) + 1
        number_label = QLabel(f"{index}.")
        type_combo = QComboBox()
        type_combo.addItem("Circular", "circular")
        type_combo.addItem("Elliptic", "elliptic")
        type_combo.addItem("Polar", "polar")
        color = DEFAULT_CURSOR_COLORS[(index - 1) % len(DEFAULT_CURSOR_COLORS)]
        color_button = ColorButton(QColor(color))

        def _spin(low, high, value, decimals=4, step=0.01):
            spin = QDoubleSpinBox()
            spin.setRange(low, high)
            spin.setDecimals(decimals)
            spin.setSingleStep(step)
            spin.setValue(value)
            spin.setMaximumWidth(70)
            return spin

        # Center fields (circular + elliptic).
        center_widget = QWidget()
        center_line = QHBoxLayout(center_widget)
        center_line.setContentsMargins(0, 0, 0, 0)
        g_spin = QLineEdit(str(g))
        g_spin.setValidator(QDoubleValidator())
        g_spin.setMaximumWidth(70)
        s_spin = QLineEdit(str(s))
        s_spin.setValidator(QDoubleValidator())
        s_spin.setMaximumWidth(70)
        radius_spin = _spin(0.0001, 2.0, radius)
        for label, widget in (
            ("G", g_spin),
            ("S", s_spin),
            ("r", radius_spin),
        ):
            center_line.addWidget(QLabel(label))
            center_line.addWidget(widget)

        # Elliptic-only fields.
        elliptic_widget = QWidget()
        elliptic_line = QHBoxLayout(elliptic_widget)
        elliptic_line.setContentsMargins(0, 0, 0, 0)
        radius_minor_spin = _spin(0.0001, 2.0, radius / 2.0)
        angle_spin = _spin(-360.0, 360.0, 0.0, decimals=1, step=5.0)
        for label, widget in (
            ("rₘ", radius_minor_spin),
            ("∠", angle_spin),
        ):
            elliptic_line.addWidget(QLabel(label))
            elliptic_line.addWidget(widget)

        # Polar-only fields.
        polar_widget = QWidget()
        polar_line = QHBoxLayout(polar_widget)
        polar_line.setContentsMargins(0, 0, 0, 0)
        phase_min_spin = _spin(-7.0, 7.0, 0.0)
        phase_max_spin = _spin(-7.0, 7.0, 1.0)
        mod_min_spin = _spin(0.0, 2.0, 0.0)
        mod_max_spin = _spin(0.0, 2.0, 1.0)
        for label, widget in (
            ("φ₋", phase_min_spin),
            ("φ₊", phase_max_spin),
            ("m₋", mod_min_spin),
            ("m₊", mod_max_spin),
        ):
            polar_line.addWidget(QLabel(label))
            polar_line.addWidget(widget)

        remove = QPushButton("✕")
        remove.setFixedWidth(28)

        row_layout.addWidget(number_label)
        row_layout.addWidget(type_combo)
        row_layout.addWidget(color_button)
        row_layout.addWidget(center_widget)
        row_layout.addWidget(elliptic_widget)
        row_layout.addWidget(polar_widget)
        row_layout.addStretch()
        row_layout.addWidget(remove)

        entry = {
            "row": frame,
            "number": number_label,
            "type": type_combo,
            "color": color_button,
            "g": g_spin,
            "s": s_spin,
            "radius": radius_spin,
            "radius_minor": radius_minor_spin,
            "angle": angle_spin,
            "phase_min": phase_min_spin,
            "phase_max": phase_max_spin,
            "mod_min": mod_min_spin,
            "mod_max": mod_max_spin,
            "center_widget": center_widget,
            "elliptic_widget": elliptic_widget,
            "polar_widget": polar_widget,
            "remove": remove,
        }
        self._cursors_layout.addWidget(frame)
        self._cursor_rows.append(entry)
        remove.clicked.connect(lambda: self._remove_cursor_row(entry))
        type_combo.currentIndexChanged.connect(
            lambda _=0, e=entry: self._on_cursor_type_changed(e)
        )
        self._on_cursor_type_changed(entry)
        self._update_cursor_remove_buttons()

    def _on_cursor_type_changed(self, entry):
        """Show only the inputs that apply to *entry*'s cursor shape."""
        cursor_type = entry["type"].currentData()
        entry["center_widget"].setVisible(cursor_type != "polar")
        entry["elliptic_widget"].setVisible(cursor_type == "elliptic")
        entry["polar_widget"].setVisible(cursor_type == "polar")

    def _remove_cursor_row(self, entry):
        """Remove *entry*'s cursor row and renumber, keeping at least one."""
        if len(self._cursor_rows) <= 1:
            return
        if entry in self._cursor_rows:
            self._cursor_rows.remove(entry)
        entry["row"].setParent(None)
        entry["row"].deleteLater()
        for i, e in enumerate(self._cursor_rows, start=1):
            e["number"].setText(f"{i}.")
        self._update_cursor_remove_buttons()

    def _update_cursor_remove_buttons(self):
        """Enable the remove buttons only while more than one cursor exists."""
        enabled = len(self._cursor_rows) > 1
        for entry in self._cursor_rows:
            entry["remove"].setEnabled(enabled)

    def _build_plot_settings_tab(self):
        """Build and return the Plot Settings tab."""
        tab = QWidget()
        outer = QVBoxLayout(tab)
        outer.addWidget(
            self._tab_header(
                "Phasor plot settings",
                "Style and export the phasor plots shared by every analysis "
                "tab. The individual (per-file) and combined (all files "
                "pooled) plots are exported and styled independently, and the "
                "chosen styles propagate to the analysis-tab overlay plots.",
            )
        )

        shared_box, shared_layout = self._section("Common appearance")
        shared_form = self._form()
        semicircle_row = QHBoxLayout()
        self._lbl_semicircle = QLabel("Semicircle (FLIM)")
        semicircle_row.addWidget(self._lbl_semicircle)
        self.plot_semicircle_checkbox = QToggleSwitch()
        self.plot_semicircle_checkbox.onColor = QColor("#27ae60")
        self.plot_semicircle_checkbox.setChecked(False)
        self.plot_semicircle_checkbox.setToolTip(
            "Draw the universal semicircle (single-exponential reference) on "
            "the phasor plot, or switch to the full polar plot."
        )
        semicircle_row.addWidget(self.plot_semicircle_checkbox)
        self._lbl_polar = QLabel("Full polar plot (HSI)")
        semicircle_row.addWidget(self._lbl_polar)
        semicircle_row.addStretch()
        shared_form.addRow(semicircle_row)

        def _update_semicircle_labels(checked):
            if checked:
                self._lbl_semicircle.setStyleSheet(
                    "font-weight: normal; color: gray;"
                )
                self._lbl_polar.setStyleSheet(
                    "font-weight: bold; color: white;"
                )
            else:
                self._lbl_semicircle.setStyleSheet(
                    "font-weight: bold; color: white;"
                )
                self._lbl_polar.setStyleSheet(
                    "font-weight: normal; color: gray;"
                )

        self.plot_semicircle_checkbox.toggled.connect(
            _update_semicircle_labels
        )
        _update_semicircle_labels(self.plot_semicircle_checkbox.isChecked())

        self.plot_white_bg_checkbox = QCheckBox("White background")
        self.plot_white_bg_checkbox.setChecked(True)
        self.plot_white_bg_checkbox.setToolTip(
            "Render the exported plots on a white background instead of the "
            "napari theme background."
        )
        shared_form.addRow(self.plot_white_bg_checkbox)

        self.plot_legend_checkbox = QCheckBox("Export legends")
        self.plot_legend_checkbox.setChecked(True)
        self.plot_legend_checkbox.setToolTip(
            "Include a legend in exported phasor plots."
        )
        shared_form.addRow(self.plot_legend_checkbox)

        self.plot_frequency_spin = QLineEdit()
        self.plot_frequency_spin.setValidator(QDoubleValidator())
        self.plot_frequency_spin.setText(str(80.0))
        self.plot_frequency_spin.setToolTip(
            "Frequency (MHz) used to place lifetime tick marks on the "
            "semicircle of the exported plot."
        )
        shared_form.addRow("Frequency (MHz):", self.plot_frequency_spin)

        center_widget = QWidget()
        center_row = QHBoxLayout(center_widget)
        center_row.setContentsMargins(0, 0, 0, 0)
        self.plot_center_checkbox = QCheckBox("Show phasor centers")
        self.plot_center_checkbox.setToolTip(
            "Mark each file's phasor center (mean G/S) on the plot."
        )
        self.plot_center_color = ColorButton(QColor("#ff1744"))
        self.plot_center_color.setToolTip("Color of the phasor-center marker.")
        center_row.addWidget(self.plot_center_checkbox)
        center_row.addWidget(QLabel("color:"))
        center_row.addWidget(self.plot_center_color)
        center_row.addStretch()
        shared_form.addRow(center_widget)

        self.plot_centers_checkbox = QCheckBox(
            "Export phasor centers (CSV, all harmonics)"
        )
        self.plot_centers_checkbox.setToolTip(
            "Write a CSV with each file's phasor center (mean G/S) for every "
            "harmonic."
        )
        shared_form.addRow(self.plot_centers_checkbox)
        shared_layout.addLayout(shared_form)
        outer.addWidget(shared_box)

        self.plot_individual_checkbox = QCheckBox(
            "Export individual phasor plots (one per file)"
        )
        self.plot_individual_checkbox.setChecked(True)
        self.plot_individual_checkbox.setToolTip(
            "Export one phasor-plot PNG per file per harmonic, for the base "
            "plot and every enabled analysis-tab overlay plot."
        )
        ind_box, ind_layout = self._section("Individual phasor plots")
        ind_layout.addWidget(self.plot_individual_checkbox)
        self._plot_individual_controls = self._plot_mode_controls("Histogram")
        ind_layout.addWidget(self._plot_individual_controls["widget"])

        self.plot_type_combo = self._plot_individual_controls["type"]
        self.plot_colormap_combo = self._plot_individual_controls["colormap"]
        self.plot_bins_spin = self._plot_individual_controls["bins"]
        self.plot_log_checkbox = self._plot_individual_controls["log"]
        self.plot_contour_widget = self._plot_individual_controls[
            "contour_widget"
        ]
        self.plot_contour_levels_spin = self._plot_individual_controls[
            "levels"
        ]
        self.plot_contour_linewidth_spin = self._plot_individual_controls[
            "linewidth"
        ]
        self.plot_individual_checkbox.toggled.connect(
            self._plot_individual_controls["widget"].setEnabled
        )
        outer.addWidget(ind_box)

        self.plot_combined_checkbox = QCheckBox(
            "Export combined phasor plot (all files pooled)"
        )
        self.plot_combined_checkbox.setChecked(True)
        self.plot_combined_checkbox.setToolTip(
            "Export combined phasor plots pooling all files, for the base "
            "plot and every enabled analysis-tab overlay plot. "
            "Histogram/Scatter pool pixels into one density; Contour can be "
            "merged into one contour or drawn per group."
        )
        comb_box, comb_layout = self._section("Combined phasor plot")
        comb_layout.addWidget(self.plot_combined_checkbox)
        self._plot_combined_controls = self._plot_mode_controls("Contour")
        comb_layout.addWidget(self._plot_combined_controls["widget"])

        self._combined_mode_row = QWidget()
        mode_row = QHBoxLayout(self._combined_mode_row)
        mode_row.setContentsMargins(0, 0, 0, 0)
        mode_row.addWidget(QLabel("Combined contour mode:"))
        self.plot_combined_mode_combo = QComboBox()
        self.plot_combined_mode_combo.addItems(["Merged", "Grouped"])
        self.plot_combined_mode_combo.setToolTip(
            "Merged pools every file into a single contour; Grouped draws one "
            "contour per file group (configure the groups below)."
        )
        stored_mode = self._group_config.get("mode", "Merged")
        self.plot_combined_mode_combo.setCurrentText(
            "Grouped" if stored_mode == "Grouped" else "Merged"
        )
        mode_row.addWidget(self.plot_combined_mode_combo)

        groups_button = QPushButton("Configure Groups…")
        groups_button.setToolTip(
            "Assign files to groups for the combined grouped contour and "
            "choose whether to show the legend."
        )
        groups_button.clicked.connect(self._open_plot_group_dialog)
        mode_row.addWidget(groups_button)
        mode_row.addStretch()
        comb_layout.addWidget(self._combined_mode_row)

        def _update_plot_groups_btn(_=None):
            is_contour = (
                self._plot_combined_controls["type"].currentData() == "Contour"
            )
            combined_on = self.plot_combined_checkbox.isChecked()
            is_grouped = (
                self.plot_combined_mode_combo.currentText() == "Grouped"
            )
            self._combined_mode_row.setVisible(is_contour)
            self.plot_combined_mode_combo.setEnabled(combined_on)
            groups_button.setVisible(is_contour and is_grouped)
            groups_button.setEnabled(is_contour and is_grouped and combined_on)

            plot_type = self._plot_combined_controls["type"].currentData()
            show_cmap = plot_type in ("Histogram", "Contour") and not (
                is_contour and is_grouped
            )
            self._plot_combined_controls["colormap"].setVisible(show_cmap)
            self._plot_combined_controls["colormap_label"].setVisible(
                show_cmap
            )

        def _on_combined_mode_changed(text):
            self._group_config["mode"] = text
            _update_plot_groups_btn()

        self.plot_combined_mode_combo.currentTextChanged.connect(
            _on_combined_mode_changed
        )
        self._plot_combined_controls["type"].currentIndexChanged.connect(
            _update_plot_groups_btn
        )
        self.plot_combined_checkbox.toggled.connect(_update_plot_groups_btn)
        self.plot_combined_checkbox.toggled.connect(
            self._plot_combined_controls["widget"].setEnabled
        )
        self.plot_combined_checkbox.toggled.connect(
            self._combined_mode_row.setEnabled
        )
        _update_plot_groups_btn()

        grp_row = QHBoxLayout()
        grp_row.addWidget(groups_button)
        grp_row.addStretch()
        comb_layout.addLayout(grp_row)
        outer.addWidget(comb_box)

        outer.addWidget(self._build_plot_zoom_box())

        outer.addStretch()
        return self._scrollable(tab)

    def _plot_mode_controls(self, default_type):
        """Build one phasor-plot style panel (individual or combined).

        Returns a dict of the created widgets plus a ``widget`` container; the
        plot type, colormap, bins, log-scale, scatter-marker and contour
        parameters can be set independently per export mode."""
        container = QWidget()
        form = self._form(container)

        type_combo = QComboBox()
        for key in ("Histogram", "Scatter", "Contour", "None"):
            type_combo.addItem(self.PLOT_TYPE_DISPLAY[key], key)
        default_index = type_combo.findData(default_type)
        if default_index >= 0:
            type_combo.setCurrentIndex(default_index)
        type_combo.setToolTip(
            "How this phasor plot is drawn: a density plot (2-D histogram), a "
            "dot plot (scatter), density contours, or None to plot no data "
            "(e.g. to export only the phasor centers)."
        )
        form.addRow("Plot type:", type_combo)

        colormap_label = QLabel("Colormap:")
        colormap_combo = self._make_colormap_combo("jet")
        colormap_combo.setToolTip("Colormap used for the phasor-plot density.")
        form.addRow(colormap_label, colormap_combo)

        bins_label = QLabel("Histogram bins:")
        bins_spin = QSpinBox()
        bins_spin.setRange(2, 2000)
        bins_spin.setValue(300)
        bins_spin.setToolTip(
            "Number of bins per axis for the histogram / contour density."
        )
        form.addRow(bins_label, bins_spin)

        log_checkbox = QCheckBox("Log scale")
        log_checkbox.setToolTip(
            "Use a logarithmic color scale for the histogram density."
        )
        form.addRow(log_checkbox)

        scatter_widget = QWidget()
        scatter_form = self._form(scatter_widget)
        marker_size_spin = QSpinBox()
        marker_size_spin.setRange(1, 200)
        marker_size_spin.setValue(5)
        marker_size_spin.setToolTip("Size of the scatter markers (points).")
        scatter_form.addRow("Marker size:", marker_size_spin)
        marker_color_button = ColorButton(QColor("#1f77b4"))
        marker_color_button.setToolTip("Color of the scatter markers.")
        scatter_form.addRow("Marker color:", marker_color_button)
        marker_alpha_spin = QDoubleSpinBox()
        marker_alpha_spin.setRange(0.01, 1.0)
        marker_alpha_spin.setSingleStep(0.05)
        marker_alpha_spin.setValue(0.3)
        marker_alpha_spin.setToolTip("Opacity of the scatter markers.")
        scatter_form.addRow("Marker alpha:", marker_alpha_spin)
        form.addRow(scatter_widget)

        contour_widget = QWidget()
        contour_form = self._form(contour_widget)
        levels_spin = QSpinBox()
        levels_spin.setRange(1, 50)
        levels_spin.setValue(7)
        levels_spin.setToolTip(
            "Number of density contour lines drawn on the plot."
        )
        contour_form.addRow("Contour levels:", levels_spin)
        linewidth_spin = QDoubleSpinBox()
        linewidth_spin.setRange(0.1, 10.0)
        linewidth_spin.setSingleStep(0.1)
        linewidth_spin.setValue(1.5)
        linewidth_spin.setToolTip("Line width of the contour lines.")
        contour_form.addRow("Contour linewidth:", linewidth_spin)
        form.addRow(contour_widget)

        def _on_type_changed(_=None):
            plot_type = type_combo.currentData()
            is_scatter = plot_type == "Scatter"
            is_contour = plot_type == "Contour"
            is_histogram = plot_type == "Histogram"

            show_density = is_histogram or is_contour
            colormap_label.setVisible(show_density)
            colormap_combo.setVisible(show_density)
            bins_label.setVisible(show_density)
            bins_spin.setVisible(show_density)

            log_checkbox.setVisible(is_histogram)
            scatter_widget.setVisible(is_scatter)
            contour_widget.setVisible(is_contour)

        type_combo.currentIndexChanged.connect(_on_type_changed)
        _on_type_changed()

        return {
            "widget": container,
            "type": type_combo,
            "colormap": colormap_combo,
            "colormap_label": colormap_label,
            "bins": bins_spin,
            "log": log_checkbox,
            "marker_size": marker_size_spin,
            "marker_color": marker_color_button,
            "marker_alpha": marker_alpha_spin,
            "scatter_widget": scatter_widget,
            "contour_widget": contour_widget,
            "levels": levels_spin,
            "linewidth": linewidth_spin,
        }

    def _build_plot_zoom_box(self):
        """Return the zoomed-section export controls."""
        box = QGroupBox("Zoomed section")
        layout = QVBoxLayout(box)

        self.plot_zoom_checkbox = QCheckBox(
            "Export a zoomed section (in addition to the full plot)"
        )
        self.plot_zoom_checkbox.setToolTip(
            "For every exported phasor plot, also save a '_zoom' PNG cropped "
            "to the limits below."
        )
        layout.addWidget(self.plot_zoom_checkbox)

        def _zoom_spin(value):
            spin = QDoubleSpinBox()
            spin.setRange(-2.0, 2.0)
            spin.setDecimals(3)
            spin.setSingleStep(0.05)
            spin.setValue(value)
            spin.setMaximumWidth(90)
            return spin

        self.plot_zoom_xmin = _zoom_spin(0.0)
        self.plot_zoom_xmax = _zoom_spin(1.0)
        self.plot_zoom_ymin = _zoom_spin(0.0)
        self.plot_zoom_ymax = _zoom_spin(0.6)

        limits_form = self._form()
        x_row = QWidget()
        x_layout = QHBoxLayout(x_row)
        x_layout.setContentsMargins(0, 0, 0, 0)
        x_layout.addWidget(QLabel("min"))
        x_layout.addWidget(self.plot_zoom_xmin)
        x_layout.addWidget(QLabel("max"))
        x_layout.addWidget(self.plot_zoom_xmax)
        x_layout.addStretch()
        limits_form.addRow("G (x) range:", x_row)

        y_row = QWidget()
        y_layout = QHBoxLayout(y_row)
        y_layout.setContentsMargins(0, 0, 0, 0)
        y_layout.addWidget(QLabel("min"))
        y_layout.addWidget(self.plot_zoom_ymin)
        y_layout.addWidget(QLabel("max"))
        y_layout.addWidget(self.plot_zoom_ymax)
        y_layout.addStretch()
        limits_form.addRow("S (y) range:", y_row)
        layout.addLayout(limits_form)

        self.plot_zoom_rect_checkbox = QCheckBox(
            "Draw the zoom region as a rectangle on the full plot"
        )
        self.plot_zoom_rect_checkbox.setToolTip(
            "Outline the zoomed region with a rectangle on the full (un-"
            "cropped) phasor plot."
        )
        layout.addWidget(self.plot_zoom_rect_checkbox)

        self._plot_zoom_limits_widgets = [
            self.plot_zoom_xmin,
            self.plot_zoom_xmax,
            self.plot_zoom_ymin,
            self.plot_zoom_ymax,
            self.plot_zoom_rect_checkbox,
        ]

        def _toggle_zoom(checked):
            for widget in self._plot_zoom_limits_widgets:
                widget.setEnabled(checked)

        self.plot_zoom_checkbox.toggled.connect(_toggle_zoom)
        _toggle_zoom(False)

        note = QLabel(
            "Applies to both the individual and combined phasor plots and to "
            "the analysis-tab overlay plots. The zoom limits are in phasor "
            "coordinates (G on x, S on y)."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(note)
        return box

    def _open_group_dialog(self):
        """Open the histogram grouping dialog and store the chosen config."""
        ext = self.format_combobox.currentData()
        files = self._scanned.get(ext, [])
        names = [os.path.basename(f) for f in files]
        dialog = HistogramSettingsDialog(
            display_mode=self._group_config.get("mode", "Merged"),
            show_sd=self._group_config.get("show_sd", True),
            central_tendency=self._group_config.get(
                "central_tendency", "None"
            ),
            show_legend=self._group_config.get("show_legend", True),
            layer_labels=names,
            group_assignments=self._group_config.get("assignments", {}),
            layer_colors=self._group_config.get("layer_colors", {}),
            group_colors=self._group_config.get("group_colors", {}),
            group_names=self._group_config.get("group_names", {}),
            parent=self,
        )
        dialog.white_bg_checkbox.setChecked(
            self._group_config.get("white_background", False)
        )
        dialog.smooth_checkbox.setChecked(
            self._group_config.get("smooth_curves", True)
        )

        if dialog.exec() == QDialog.Accepted:
            self._group_config.update(
                {
                    "mode": dialog.mode_combo.currentText(),
                    "assignments": dialog.get_group_assignments(),
                    "group_names": dialog.get_group_names(),
                    "group_colors": dialog.get_group_colors(),
                    "layer_colors": dialog.get_layer_colors(),
                    "show_sd": dialog.sd_checkbox.isChecked(),
                    "central_tendency": (
                        dialog.central_tendency_combo.currentText()
                    ),
                    "show_legend": dialog.legend_checkbox.isChecked(),
                    "white_background": dialog.white_bg_checkbox.isChecked(),
                    "smooth_curves": dialog.smooth_checkbox.isChecked(),
                }
            )

    def _open_plot_group_dialog(self):
        """Assign files to groups for the combined grouped contour."""
        from .plotter import ContourLayerSettingsDialog

        ext = self.format_combobox.currentData()
        files = self._scanned.get(ext, [])
        names = [os.path.basename(f) for f in files]
        cfg = self._group_config
        dialog = ContourLayerSettingsDialog(
            groups_only=True,
            display_mode="Grouped",
            show_legend=cfg.get("show_legend", True),
            merged_colormap=cfg.get("contour_merged_colormap", "jet"),
            layer_labels=names,
            group_assignments=cfg.get("assignments", {}),
            group_colors=cfg.get("group_colors", {}),
            group_names=cfg.get("group_names", {}),
            layer_styles=cfg.get("contour_layer_styles", {}),
            group_styles=cfg.get("contour_group_styles", {}),
            parent=self,
        )
        if dialog.exec() == QDialog.Accepted:
            cfg.update(
                {
                    "mode": "Grouped",
                    "assignments": dialog.get_group_assignments(),
                    "group_names": dialog.get_group_names(),
                    "group_colors": dialog.get_group_colors(),
                    "show_legend": dialog.get_show_legend(),
                    "contour_group_styles": dialog.get_group_styles(),
                }
            )

    def _group_for(self, filename):
        """Return ``(group_id, group_name, color)`` for ``filename``."""
        config = self._group_config
        mode = config.get("mode", "Merged")
        if mode == "Merged":
            return (0, "All", DEFAULT_CURSOR_COLORS[0])
        if mode == "Individual layers":
            color = config.get("layer_colors", {}).get(filename, None)
            return (filename, filename, color)
        gid = config.get("assignments", {}).get(filename, 1)
        name = config.get("group_names", {}).get(gid, f"Group {gid}")
        color = config.get("group_colors", {}).get(
            gid, DEFAULT_CURSOR_COLORS[(gid - 1) % len(DEFAULT_CURSOR_COLORS)]
        )
        return (gid, name, color)

    def _build_run_footer(self, layout):
        """Add the run button and progress footer to *layout*."""
        self.run_button = QPushButton("Run batch analysis")
        self.run_button.clicked.connect(self.run_batch)
        layout.addWidget(self.run_button)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

    def _add_component_row(self, name="", g=0.0, s=0.0):
        """Append a component row at phasor coordinates (*g*, *s*)."""
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)

        index = len(self._component_rows) + 1
        number_label = QLabel(f"{index}.")
        name_edit = QLineEdit(name or f"Component {index}")
        name_edit.setPlaceholderText("name")

        g_spin = QLineEdit(str(g))
        g_spin.setValidator(QDoubleValidator())
        s_spin = QLineEdit(str(s))
        s_spin.setValidator(QDoubleValidator())

        lifetime_edit = QLineEdit()

        coords = {self._component_current_harmonic: (float(g), float(s))}
        lifetime_edit.setPlaceholderText("τ (ns)")
        lifetime_edit.setMaximumWidth(70)
        lifetime_edit.setToolTip(
            "Type a lifetime (ns) and press Enter to set G/S from it."
        )

        colormap_combo = self._make_colormap_combo("jet")
        colormap_combo.setMaximumWidth(120)
        colormap_combo.setToolTip("Colormap for this component's fraction.")

        remove = QPushButton("✕")
        remove.setFixedWidth(28)

        row_layout.addWidget(number_label)
        row_layout.addWidget(name_edit, 1)
        row_layout.addWidget(QLabel("G"))
        row_layout.addWidget(g_spin)
        row_layout.addWidget(QLabel("S"))
        row_layout.addWidget(s_spin)
        row_layout.addWidget(QLabel("τ"))
        row_layout.addWidget(lifetime_edit)
        row_layout.addWidget(colormap_combo)
        row_layout.addWidget(remove)

        entry = {
            "row": row,
            "number": number_label,
            "name": name_edit,
            "g": g_spin,
            "s": s_spin,
            "lifetime": lifetime_edit,
            "colormap": colormap_combo,
            "remove": remove,
            "coords": coords,
        }
        self._components_layout.addWidget(row)
        self._component_rows.append(entry)
        remove.clicked.connect(lambda: self._remove_component_row(entry))
        lifetime_edit.editingFinished.connect(
            lambda e=entry: self._on_component_lifetime_edited(e)
        )

        g_spin.editingFinished.connect(
            lambda e=entry: self._store_component_coord(e)
        )
        s_spin.editingFinished.connect(
            lambda e=entry: self._store_component_coord(e)
        )
        self._update_component_controls()

    def _on_component_lifetime_edited(self, entry):
        """Set a component's G/S from a typed lifetime (ns)."""
        text = entry["lifetime"].text().strip()
        if not text:
            return
        try:
            lifetime = float(text)
        except ValueError:
            return
        frequency = float(self.components_frequency_spin.text() or 0.0)
        harmonic = self.components_harmonic_spin.value()
        if frequency <= 0:
            return
        real, imag = phasor_from_lifetime(frequency * harmonic, lifetime)
        entry["g"].setText(str(float(real)))
        entry["s"].setText(str(float(imag)))
        self._store_component_coord(entry)

    @staticmethod
    def _parse_float(text):
        """Return ``float(text)`` or ``None`` if blank/invalid."""
        text = (text or "").strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    def _store_component_coord(self, entry):
        """Persist a row's visible G/S into the current harmonic's store."""
        g = self._parse_float(entry["g"].text())
        s = self._parse_float(entry["s"].text())
        harmonic = self._component_current_harmonic
        if g is None and s is None:
            entry["coords"].pop(harmonic, None)
        else:
            entry["coords"][harmonic] = (g or 0.0, s or 0.0)

    def _sync_component_coords(self):
        """Commit every row's visible G/S into the current harmonic store."""
        for entry in self._component_rows:
            self._store_component_coord(entry)

    def _load_component_coords(self):
        """Show each row's stored G/S for the current editing harmonic."""
        harmonic = self._component_current_harmonic
        for entry in self._component_rows:
            stored = entry["coords"].get(harmonic)
            for key, value in zip(
                ("g", "s"), stored if stored else (None, None), strict=False
            ):
                widget = entry[key]
                widget.blockSignals(True)
                widget.setText("" if value is None else str(value))
                widget.blockSignals(False)
            entry["lifetime"].blockSignals(True)
            entry["lifetime"].clear()
            entry["lifetime"].blockSignals(False)

    def _on_component_harmonic_changed(self, value):
        """Swap the visible G/S fields to a different harmonic's locations."""
        self._sync_component_coords()
        self._component_current_harmonic = value
        self._load_component_coords()
        self._update_component_controls()

    def _active_component_harmonics(self):
        """Return harmonics that have G/S for *every* component row, sorted."""
        if not self._component_rows:
            return []
        common = set(self._component_rows[0]["coords"])
        for entry in self._component_rows[1:]:
            common &= set(entry["coords"])
        return sorted(common)

    def _remove_component_row(self, entry):
        """Remove *entry*'s component row, keeping the two-component minimum."""
        if len(self._component_rows) <= 2:
            return
        if entry in self._component_rows:
            self._component_rows.remove(entry)
        entry["row"].setParent(None)
        entry["row"].deleteLater()
        for i, e in enumerate(self._component_rows, start=1):
            e["number"].setText(f"{i}.")
        self._update_component_controls()

    def _set_component_rows(self, parsed):
        """Replace all rows with ``parsed`` ``(name, g, s)`` tuples."""
        for entry in list(self._component_rows):
            entry["row"].setParent(None)
            entry["row"].deleteLater()
        self._component_rows = []

        self.components_harmonic_spin.blockSignals(True)
        self.components_harmonic_spin.setValue(1)
        self.components_harmonic_spin.blockSignals(False)
        self._component_current_harmonic = 1
        for name, g, s in parsed:
            self._add_component_row(name, g, s)
        self._update_component_controls()

    def _set_component_harmonic_coords(self, per_harmonic):
        """Seed component rows with G/S at additional harmonics."""
        for harmonic, coords in per_harmonic.items():
            if int(harmonic) == 1:
                continue
            for entry, gs in zip(self._component_rows, coords, strict=False):
                if gs is None:
                    continue
                entry["coords"][int(harmonic)] = (
                    float(gs[0]),
                    float(gs[1]),
                )
        self._update_component_note()

    def _update_component_controls(self):
        """Re-sync the component controls to the row count and analysis type."""
        count = len(self._component_rows)
        for entry in self._component_rows:
            entry["remove"].setEnabled(count > 2)

        current = self.analysis_type_combo.currentText()
        self.analysis_type_combo.blockSignals(True)
        self.analysis_type_combo.clear()
        if count == 2:
            self.analysis_type_combo.addItems(
                ["Linear Projection", "Component Fit"]
            )
        else:
            self.analysis_type_combo.addItems(["Component Fit"])
        index = self.analysis_type_combo.findText(current)
        if index >= 0:
            self.analysis_type_combo.setCurrentIndex(index)
        self.analysis_type_combo.blockSignals(False)

        self._update_component_note()

    def _update_component_note(self):
        """Refresh the components guidance note (incl. multi-harmonic state)."""
        count = len(self._component_rows)
        required = required_component_harmonics(count)
        if required <= 1:
            self.components_note.setText(
                "Linear Projection (2 components) outputs one fraction image; "
                "Component Fit outputs one fraction image per component."
            )
            self.components_note.setStyleSheet("color: gray; font-size: 11px;")
            return

        active = self._active_component_harmonics()
        have = len(active)
        message = (
            f"Fitting {count} components needs locations at {required} "
            f"harmonics. Set G/S for each component, then change the "
            f"'Harmonic' selector and set their locations at the other "
            f"harmonics. Harmonics with all locations: "
            f"{active if active else 'none'}."
        )
        if have >= required:
            self.components_note.setText("✓ " + message)
            self.components_note.setStyleSheet(
                "color: #27ae60; font-size: 11px;"
            )
        else:
            self.components_note.setText("⚠ " + message)
            self.components_note.setStyleSheet(
                "color: #d35400; font-size: 11px;"
            )

    def _open_component_line_style_dialog(self):
        """Edit the component dot/line style used in the exported plot."""
        style = self._component_line_style
        dialog = QDialog(self)
        dialog.setWindowTitle("Component Line Settings")
        vbox = QVBoxLayout(dialog)

        row1 = QHBoxLayout()
        colormap_cb = QCheckBox("Overlay colormap line")
        colormap_cb.setChecked(style["show_colormap_line"])
        row1.addWidget(colormap_cb)
        dots_cb = QCheckBox("Show component positions")
        dots_cb.setChecked(style["show_component_dots"])
        row1.addWidget(dots_cb)
        row1.addStretch()
        vbox.addLayout(row1)

        form = QFormLayout()
        offset_spin = QDoubleSpinBox()
        offset_spin.setRange(-0.5, 0.5)
        offset_spin.setSingleStep(0.005)
        offset_spin.setDecimals(3)
        offset_spin.setValue(style["line_offset"])
        form.addRow("Line offset:", offset_spin)

        width_spin = QDoubleSpinBox()
        width_spin.setRange(0.5, 20.0)
        width_spin.setSingleStep(0.5)
        width_spin.setValue(style["line_width"])
        form.addRow("Line width:", width_spin)

        alpha_spin = QDoubleSpinBox()
        alpha_spin.setRange(0.0, 1.0)
        alpha_spin.setSingleStep(0.05)
        alpha_spin.setDecimals(2)
        alpha_spin.setValue(style["line_alpha"])
        form.addRow("Line alpha:", alpha_spin)

        gamma_spin = QDoubleSpinBox()
        gamma_spin.setRange(0.01, 10.0)
        gamma_spin.setSingleStep(0.05)
        gamma_spin.setDecimals(2)
        gamma_spin.setValue(style.get("colormap_gamma", 1.0))
        gamma_spin.setToolTip(
            "Power-law gamma applied to the colormap line and fraction "
            "histogram gradient, matching the fraction layer's gamma."
        )
        form.addRow("Colormap gamma:", gamma_spin)

        color_button = ColorButton(QColor(style["default_component_color"]))
        color_row = QHBoxLayout()
        color_row.addWidget(color_button)
        color_row.addStretch()
        color_container = QWidget()
        color_container.setLayout(color_row)
        form.addRow("Default line color:", color_container)
        vbox.addLayout(form)

        hist_group = QGroupBox("Fraction histogram overlay")
        hist_form = QFormLayout(hist_group)
        hist_cb = QCheckBox("Overlay fraction histogram on the line")
        hist_cb.setChecked(style.get("show_fraction_histogram", False))
        hist_cb.setToolTip(
            "Overlay the first component's fraction histogram along the line "
            "joining the two components, colored with the line's colormap. "
            "Applies to a two-component Linear Projection."
        )
        hist_form.addRow(hist_cb)

        hist_height_spin = QDoubleSpinBox()
        hist_height_spin.setRange(0.05, 1.0)
        hist_height_spin.setSingleStep(0.05)
        hist_height_spin.setDecimals(2)
        hist_height_spin.setValue(style.get("histogram_overlay_height", 0.3))
        hist_form.addRow("Histogram height:", hist_height_spin)

        hist_offset_spin = QDoubleSpinBox()
        hist_offset_spin.setRange(-1.0, 1.0)
        hist_offset_spin.setSingleStep(0.01)
        hist_offset_spin.setDecimals(3)
        hist_offset_spin.setValue(style.get("histogram_offset", 0.0))
        hist_offset_spin.setToolTip(
            "Shift the histogram relative to the line. Positive keeps it on "
            "one side; negative flips it to the other side."
        )
        hist_form.addRow("Histogram offset:", hist_offset_spin)

        hist_transp_spin = QDoubleSpinBox()
        hist_transp_spin.setRange(0.0, 1.0)
        hist_transp_spin.setSingleStep(0.05)
        hist_transp_spin.setDecimals(2)
        hist_transp_spin.setValue(1.0 - style.get("histogram_alpha", 0.75))
        hist_form.addRow("Histogram transparency:", hist_transp_spin)
        vbox.addWidget(hist_group)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        vbox.addWidget(buttons)

        if dialog.exec() == QDialog.Accepted:
            self._component_line_style = {
                "show_colormap_line": colormap_cb.isChecked(),
                "show_component_dots": dots_cb.isChecked(),
                "line_offset": offset_spin.value(),
                "line_width": width_spin.value(),
                "line_alpha": alpha_spin.value(),
                "colormap_gamma": gamma_spin.value(),
                "default_component_color": color_button.color().name(),
                "show_fraction_histogram": hist_cb.isChecked(),
                "histogram_overlay_height": hist_height_spin.value(),
                "histogram_offset": hist_offset_spin.value(),
                "histogram_alpha": 1.0 - hist_transp_spin.value(),
            }

    def _open_component_label_style_dialog(self):
        """Edit the component label style used in the exported plot."""
        style = self._component_label_style
        dialog = QDialog(self)
        dialog.setWindowTitle("Component Label Style")
        vbox = QVBoxLayout(dialog)

        show_labels_cb = QCheckBox("Show component name labels on phasor plot")
        show_labels_cb.setChecked(style.get("show_labels", False))
        show_labels_cb.setToolTip(
            "When checked, each component's name is drawn next to its location "
            "on the exported phasor plot."
        )
        vbox.addWidget(show_labels_cb)

        row = QHBoxLayout()
        row.addWidget(QLabel("Size:"))
        fontsize_spin = QSpinBox()
        fontsize_spin.setRange(6, 72)
        fontsize_spin.setValue(style["fontsize"])
        row.addWidget(fontsize_spin)
        bold_cb = QCheckBox("Bold")
        bold_cb.setChecked(style["bold"])
        row.addWidget(bold_cb)
        italic_cb = QCheckBox("Italic")
        italic_cb.setChecked(style["italic"])
        row.addWidget(italic_cb)
        color_button = ColorButton(QColor(style["color"]))
        row.addWidget(QLabel("Color:"))
        row.addWidget(color_button)
        row.addStretch()
        vbox.addLayout(row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        vbox.addWidget(buttons)

        if dialog.exec() == QDialog.Accepted:
            self._component_label_style = {
                "show_labels": show_labels_cb.isChecked(),
                "fontsize": fontsize_spin.value(),
                "bold": bold_cb.isChecked(),
                "italic": italic_cb.isChecked(),
                "color": color_button.color().name(),
            }

    def _phasor_layer_names(self):
        """Return the names of the viewer's layers holding phasor data."""
        return [
            layer.name
            for layer in self.viewer.layers
            if isinstance(layer, Image)
            and all(
                key in layer.metadata
                for key in ["G", "S", "G_original", "S_original"]
            )
        ]

    def _on_layers_changed(self, event=None):
        """Refresh layer comboboxes when the viewer's layer list changes."""
        self._populate_layer_comboboxes()

    def closeEvent(self, event):
        """Disconnect viewer events so teardown can't fire into a freed widget."""
        with contextlib.suppress(TypeError, ValueError, AttributeError):
            self.viewer.layers.events.inserted.disconnect(
                self._on_layers_changed
            )
        with contextlib.suppress(TypeError, ValueError, AttributeError):
            self.viewer.layers.events.removed.disconnect(
                self._on_layers_changed
            )
        event.accept()

    def _populate_layer_comboboxes(self):
        """Refill the reference layer combobox, keeping any valid selection."""
        names = self._phasor_layer_names()
        combo = getattr(self, "calib_reference_combo", None)
        if combo is None:
            return
        current = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(names)
        if current in names:
            combo.setCurrentText(current)
        combo.blockSignals(False)

    def _on_select_folder(self):
        """Prompt for the input folder and rescan it."""
        folder = QFileDialog.getExistingDirectory(self, "Select input folder")
        if not folder:
            return
        self._input_folder = folder
        self.folder_label.setText(folder)
        self._rescan()

    def _rescan(self):
        """Rescan the input folder and repopulate the format combobox."""
        if not self._input_folder:
            return
        self._scanned = scan_folder(
            self._input_folder, self.subfolders_checkbox.isChecked()
        )

        self._signal_capable_cache = {}
        self.format_combobox.blockSignals(True)
        self.format_combobox.clear()
        for ext in sorted(self._scanned):
            count = len(self._scanned[ext])
            self.format_combobox.addItem(f"{ext} ({count} files)", ext)
        self.format_combobox.blockSignals(False)
        if self.format_combobox.count():
            self.format_combobox.setCurrentIndex(0)
            self._on_format_changed()
        else:
            self.status_label.setText(
                "No supported files found in the selected folder."
            )
        self._refresh_signal_availability()

    def _on_format_changed(self):
        """Rebuild the reader options and dependent rows for the new format."""
        ext = self.format_combobox.currentData()
        if ext:
            self.read_options_widget.set_extension(ext)
        if (
            getattr(self, "calib_source_combo", None) is not None
            and self.calib_source_combo.currentData() == "subfolder"
        ):
            self._rebuild_subfolder_rows()
        if getattr(self, "_mask_rows_layout", None) is not None:
            self._rebuild_mask_rows()
        self._refresh_signal_availability()

    def _on_calib_source_changed(self):
        """Show only the settings belonging to the chosen calibration source."""
        source = self.calib_source_combo.currentData()
        self.calib_same_widget.setVisible(source == "same")
        self.calib_subfolder_widget.setVisible(source == "subfolder")
        self.calib_copied_label.setVisible(source == "copied")
        self._calib_freq_lifetime_widget.setEnabled(source != "copied")
        if source == "subfolder":
            self._rebuild_subfolder_rows()

    def _subfolder_key(self, path):
        """Return the top-level subfolder of ``path`` relative to the input."""
        if not self._input_folder:
            return ""
        rel = os.path.relpath(os.path.dirname(path), self._input_folder)
        if rel in (".", ""):
            return ""
        return rel.split(os.sep)[0]

    def _subfolder_keys(self):
        """Return the sorted subfolders holding files of the chosen format."""
        ext = self.format_combobox.currentData()
        files = self._scanned.get(ext, [])
        return sorted({self._subfolder_key(f) for f in files})

    def _rebuild_subfolder_rows(self):
        """Rebuild one reference-file row per subfolder of the input folder."""
        layout = self.calib_subfolder_rows_layout
        for i in reversed(range(layout.count())):
            widget = layout.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self._subfolder_ref_edits = {}

        keys = self._subfolder_keys()
        if not keys:
            placeholder = QLabel(
                "Scan a folder and select a format to list subfolders."
            )
            placeholder.setStyleSheet("color: gray;")
            layout.addWidget(placeholder)
            return

        for key in keys:
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            label = QLabel(key or "(root)")
            label.setMinimumWidth(90)
            edit = QLineEdit()
            edit.setPlaceholderText("OME-TIFF reference…")
            browse = QPushButton("Browse")
            browse.clicked.connect(
                lambda _=False, e=edit: self._browse_subfolder_ref(e)
            )
            row_layout.addWidget(label)
            row_layout.addWidget(edit, 1)
            row_layout.addWidget(browse)
            layout.addWidget(row)
            self._subfolder_ref_edits[key] = edit

    def _supported_files_filter(self):
        """Return a Qt file-dialog filter for all supported file formats."""
        patterns = " ".join(f"*{ext}" for ext in supported_extensions())
        return f"Supported files ({patterns});;All files (*)"

    def _browse_subfolder_ref(self, edit):
        """Prompt for a subfolder's reference file and write it into *edit*."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select reference file",
            "",
            self._supported_files_filter(),
        )
        if path:
            edit.setText(path)

    def _on_browse_calibration_file(self):
        """Prompt for the calibration reference file."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select reference file",
            "",
            self._supported_files_filter(),
        )
        if path:
            self.calib_file_edit.setText(path)

    def _on_select_export(self):
        """Prompt for the export folder and re-check whether the run can start."""
        folder = QFileDialog.getExistingDirectory(self, "Select export folder")
        if folder:
            self._export_folder = folder
            self.export_folder_label.setText(folder)
            self._update_run_enabled()

    def _export_dpi(self):
        """Return the DPI selected in the Setup tab for exported images."""
        return self.export_dpi_combo.currentData()

    def _on_copy_settings(self):
        """Copy analysis settings from a chosen layer or OME-TIFF into the tabs."""
        dialog = CopySettingsDialog(self._phasor_layer_names(), self)
        if dialog.exec_() != QDialog.Accepted:
            return
        settings = None
        file_path = dialog.selected_file()
        if file_path:
            try:
                settings = read_ome_tiff_settings(file_path)
            except Exception as exc:  # noqa: BLE001
                show_error(f"Could not read settings: {exc}")
                return
        else:
            name = dialog.selected_layer()
            if name and name in self.viewer.layers:
                layer_metadata = self.viewer.layers[name].metadata
                settings = dict(layer_metadata.get("settings", {}))

                if layer_metadata.get("harmonics") is not None:
                    settings.setdefault(
                        "harmonics", layer_metadata["harmonics"]
                    )
        if not settings:
            show_error("No settings found in the selected source.")
            return
        self._apply_settings_to_ui(settings)
        show_info("Settings copied. Review and adjust before running.")

    def _has_extra_outputs(self):
        """Whether any non-layer output (plots/per-tab) is requested."""
        if (
            self.plot_individual_checkbox.isChecked()
            or self.plot_combined_checkbox.isChecked()
        ):
            return True
        if self._signal_export_requested():
            return True
        plot_toggles = [
            (self.components_group, self.components_plot_toggle),
            (self.mapping_group, self.mapping_plot_toggle),
            (self.fret_group, self.fret_plot_toggle),
            (self.selection_group, self.selection_plot_toggle),
        ]
        for group, toggle in plot_toggles:
            if group.isChecked() and toggle.isChecked():
                return True
        if self.selection_group.isChecked() and (
            self.selection_stats_checkbox.isChecked()
            or self.selection_export_combo.checkedItems()
        ):
            return True
        export_specs = [
            (self.components_group, self.components_export_controls),
            (self.mapping_group, self.mapping_export_controls),
            (self.fret_group, self.fret_export_controls),
        ]
        for group, controls in export_specs:
            if group.isChecked() and (
                controls["stats"].isChecked()
                or controls["histogram"].checkedItems()
                or controls["image"].checkedItems()
            ):
                return True
        return False

    def _update_run_enabled(self):
        """Enable the run button only once the batch is fully configured."""
        has_files = self.format_combobox.count() > 0
        has_export = bool(self._export_folder)
        has_type = (
            any(
                checkbox.isChecked()
                for checkbox in (
                    self.export_ometiff_checkbox,
                    self.export_csv_checkbox,
                    self.export_image_checkbox,
                )
            )
            or self._has_extra_outputs()
        )
        self.run_button.setEnabled(has_files and has_export and has_type)
        if not has_files:
            tip = "Select an input folder containing supported files."
        elif not has_export:
            tip = "Select an export folder."
        elif not has_type:
            tip = "Select an export format or enable an analysis output."
        else:
            tip = "Run the batch analysis."
        self.run_button.setToolTip(tip)

    def _apply_settings_to_ui(self, settings):
        """Populate the analysis tabs from a settings dict."""
        if not settings:
            return

        frequency = settings.get("frequency")
        if frequency is not None:
            try:
                value = float(
                    np.atleast_1d(np.asarray(frequency, dtype=float)).ravel()[
                        0
                    ]
                )
                self.calib_frequency_spin.setText(str(value))
                self.components_frequency_spin.setText(str(value))
                self.mapping_frequency_spin.setText(str(value))
                self.fret_frequency_spin.setText(str(value))
                self.plot_frequency_spin.setText(str(value))
            except (TypeError, ValueError):
                pass

        if (
            settings.get("calibrated")
            and "calibration_phase" in settings
            and "calibration_modulation" in settings
        ):
            self._copied_calibration = {
                "phi_zero": np.asarray(settings["calibration_phase"]),
                "mod_zero": np.asarray(settings["calibration_modulation"]),
                "harmonics": settings.get("harmonics"),
            }
            self.calibration_group.setChecked(True)
            index = self.calib_source_combo.findData("copied")
            if index >= 0:
                self.calib_source_combo.setCurrentIndex(index)

        self._apply_filter_settings_to_ui(settings)
        self._apply_component_settings_to_ui(settings)
        self._apply_group_config_from_settings(settings)
        self._apply_plot_settings_to_ui(settings)
        self._apply_mapping_settings_to_ui(settings)
        self._apply_selection_settings_to_ui(settings)
        self._apply_fret_settings_to_ui(settings)

    def _apply_plot_settings_to_ui(self, settings):
        """Apply copied plot *settings* to the Plot Settings tab."""
        if "semi_circle" in settings:
            self.plot_semicircle_checkbox.setChecked(
                not settings["semi_circle"]
            )
        if "log_scale" in settings:
            self.plot_log_checkbox.setChecked(settings["log_scale"])
        if "colormap" in settings:
            index = self.plot_colormap_combo.findText(settings["colormap"])
            if index >= 0:
                self.plot_colormap_combo.setCurrentIndex(index)
        if "bins" in settings:
            self.plot_bins_spin.setValue(int(settings["bins"]))

    def _apply_mapping_settings_to_ui(self, settings):
        """Apply copied lifetime/mapping *settings* to the Phasor Mapping tab."""
        lifetime = settings.get("lifetime")
        if not lifetime:
            return
        self.mapping_group.setChecked(True)
        if "lifetime_type" in lifetime:
            self.mapping_output_combo.deselectAll()
            for index in range(self.mapping_output_combo.model().rowCount()):
                item = self.mapping_output_combo.model().item(index)
                if item and item.text() == lifetime["lifetime_type"]:
                    item.setCheckState(Qt.Checked)
                    break

    def _apply_selection_settings_to_ui(self, settings):
        """Populate the Selection tab from persisted manual cursors."""
        selections = settings.get("selections") or {}
        circular = selections.get("circular_cursors") or []
        elliptical = selections.get("elliptical_cursors") or []
        polar = selections.get("polar_cursors") or []
        if not (circular or elliptical or polar):
            return

        self.selection_group.setChecked(True)
        self.selection_mode_combo.setCurrentText("Manual cursors")
        for entry in list(self._cursor_rows):
            entry["row"].setParent(None)
            entry["row"].deleteLater()
        self._cursor_rows = []

        def _new_row(cursor_type):
            self._add_cursor_row()
            entry = self._cursor_rows[-1]
            index = entry["type"].findData(cursor_type)
            if index >= 0:
                entry["type"].setCurrentIndex(index)
            return entry

        def _set_color(entry, color):
            if color is None:
                return
            try:
                if isinstance(color, (tuple, list)):
                    entry["color"].set_color(
                        QColor(*[int(round(c)) for c in color])
                    )
                else:
                    entry["color"].set_color(QColor(color))
            except (TypeError, ValueError):
                pass

        for c in circular:
            entry = _new_row("circular")
            entry["g"].setText(str(float(c.get("g", 0.0))))
            entry["s"].setText(str(float(c.get("s", 0.0))))
            if "radius" in c:
                entry["radius"].setValue(float(c["radius"]))
            _set_color(entry, c.get("color"))

        for c in elliptical:
            entry = _new_row("elliptic")
            entry["g"].setText(str(float(c.get("g", 0.0))))
            entry["s"].setText(str(float(c.get("s", 0.0))))
            if "radius" in c:
                entry["radius"].setValue(float(c["radius"]))
            if "radius_minor" in c:
                entry["radius_minor"].setValue(float(c["radius_minor"]))
            if "angle" in c:
                entry["angle"].setValue(float(c["angle"]))
            _set_color(entry, c.get("color"))

        for c in polar:
            entry = _new_row("polar")
            if "phase_min" in c:
                entry["phase_min"].setValue(float(c["phase_min"]))
            if "phase_max" in c:
                entry["phase_max"].setValue(float(c["phase_max"]))
            if "modulation_min" in c:
                entry["mod_min"].setValue(float(c["modulation_min"]))
            if "modulation_max" in c:
                entry["mod_max"].setValue(float(c["modulation_max"]))
            _set_color(entry, c.get("color"))

    def _apply_fret_settings_to_ui(self, settings):
        """Apply copied FRET *settings* to the FRET tab."""
        fret = settings.get("fret_analysis")
        if not fret:
            return
        self.fret_group.setChecked(True)
        if "donor_lifetime" in fret:
            self.fret_donor_lifetime_spin.setValue(
                float(fret["donor_lifetime"])
            )
        if "donor_background" in fret:
            self.fret_background_spin.setValue(float(fret["donor_background"]))
        if "donor_fretting" in fret:
            self.fret_fretting_spin.setValue(float(fret["donor_fretting"]))
        if "background_real" in fret:
            self.fret_bg_real_spin.setValue(float(fret["background_real"]))
        if "background_imag" in fret:
            self.fret_bg_imag_spin.setValue(float(fret["background_imag"]))

    def _apply_group_config_from_settings(self, settings):
        """Restore the histogram grouping config from copied *settings*."""
        stored = settings.get("batch_group_config")
        if not stored:
            return
        self._group_config = {
            "mode": stored.get("mode", "Merged"),
            "assignments": dict(stored.get("assignments", {})),
            "group_names": {
                int(k): v for k, v in stored.get("group_names", {}).items()
            },
            "group_colors": {
                int(k): v for k, v in stored.get("group_colors", {}).items()
            },
            "show_sd": stored.get("show_sd", True),
            "central_tendency": stored.get("central_tendency", "None"),
            "show_legend": stored.get("show_legend", True),
        }

    def _apply_filter_settings_to_ui(self, settings):
        """Apply copied filter and threshold *settings* to the Filter tab."""
        filter_settings = settings.get("filter") or {}
        method = filter_settings.get("method")
        enabled = False
        if method:
            enabled = True
            method = str(method).lower()
            if method == "median":
                self.filter_method_combo.setCurrentText("Median")
                if filter_settings.get("size") is not None:
                    self.median_size_spin.setValue(
                        int(filter_settings["size"])
                    )
                if filter_settings.get("repeat") is not None:
                    self.median_repeat_spin.setValue(
                        int(filter_settings["repeat"])
                    )
            elif method == "wavelet":
                self.filter_method_combo.setCurrentText(
                    "Wavelet (binlet pawFLIM)"
                )
                if filter_settings.get("sigma") is not None:
                    self.wavelet_sigma_spin.setValue(
                        float(filter_settings["sigma"])
                    )
                if filter_settings.get("levels") is not None:
                    self.wavelet_levels_spin.setValue(
                        int(filter_settings["levels"])
                    )

        threshold_method = settings.get("threshold_method")
        if threshold_method:
            enabled = True
            self.threshold_method_combo.setCurrentText(
                str(threshold_method).capitalize()
            )
            if settings.get("threshold") is not None:
                self.threshold_min_spin.setValue(float(settings["threshold"]))
            if settings.get("threshold_upper") is not None:
                self.threshold_max_spin.setValue(
                    float(settings["threshold_upper"])
                )

        if enabled:
            self.filter_group.setChecked(True)

    def _apply_component_settings_to_ui(self, settings):
        """Apply copied component *settings* to the Components tab."""
        component_analysis = settings.get("component_analysis") or {}
        line_settings = (
            component_analysis.get("two_component_line_settings")
            or component_analysis.get("line_settings")
            or {}
        )
        for key in (
            "show_colormap_line",
            "show_component_dots",
            "line_offset",
            "line_width",
            "line_alpha",
            "default_component_color",
            "show_fraction_histogram",
            "histogram_overlay_height",
            "histogram_offset",
            "histogram_alpha",
        ):
            if key in line_settings:
                self._component_line_style[key] = line_settings[key]

        label_settings = (
            component_analysis.get("two_components_label_settings")
            or component_analysis.get("label_settings")
            or {}
        )
        for key in ("fontsize", "bold", "italic", "color"):
            if key in label_settings:
                self._component_label_style[key] = label_settings[key]

        components = component_analysis.get("components") or {}
        if not components:
            return
        try:
            items = sorted(components.items(), key=lambda kv: int(kv[0]))
        except (ValueError, TypeError):
            items = list(components.items())

        parsed = []

        per_harmonic = {}
        for _key, comp_data in items:
            gs_harmonics = (comp_data or {}).get("gs_harmonics") or {}
            harmonic_data = gs_harmonics.get("1")
            if harmonic_data is None and gs_harmonics:
                harmonic_data = next(iter(gs_harmonics.values()))
            if not harmonic_data or "g" not in harmonic_data:
                continue
            name = comp_data.get("name") or f"Component {len(parsed) + 1}"
            parsed.append(
                (name, float(harmonic_data["g"]), float(harmonic_data["s"]))
            )
            row_index = len(parsed) - 1
            for harmonic_key, data in gs_harmonics.items():
                if not data or "g" not in data or "s" not in data:
                    continue
                slot = per_harmonic.setdefault(harmonic_key, [])
                while len(slot) <= row_index:
                    slot.append(None)
                slot[row_index] = (float(data["g"]), float(data["s"]))

        if len(parsed) < 2:
            return
        self._set_component_rows(parsed)
        self._set_component_harmonic_coords(per_harmonic)
        self.components_group.setChecked(True)

        analysis_type = component_analysis.get("analysis_type")
        if analysis_type:
            index = self.analysis_type_combo.findText(analysis_type)
            if index >= 0:
                self.analysis_type_combo.setCurrentIndex(index)

    def _load_reference_layer(self, path, harmonics):
        """Load the first phasor layer from a file as an in-memory Image."""
        reader = napari_get_reader(
            path, reader_options={}, harmonics=harmonics
        )
        if reader is None:
            raise ValueError(f"No reader for reference file: {path}")
        layer_data = reader(path)
        data, add_kw = layer_data[0][0], layer_data[0][1]
        return Image(
            data,
            name=add_kw.get("name", os.path.basename(path)),
            metadata=add_kw.get("metadata", {}),
        )

    def build_pipeline(self, harmonics):
        """Resolve the configured filter/component steps into a pipeline."""
        pipeline = BatchPipeline()
        if self.filter_group.isChecked():
            pipeline.filter = self._collect_filter_kwargs()
        if self.components_group.isChecked():
            pipeline.components = self._collect_components(harmonics)
        if self.mapping_group.isChecked():
            pipeline.mapping = self._collect_mapping()
        if self.fret_group.isChecked():
            pipeline.fret = self._collect_fret()
        if self.selection_group.isChecked():
            pipeline.selection = self._collect_selection()
        return pipeline

    def _collect_mapping(self):
        """Return the Phasor Mapping tab's settings as a dict for the run."""
        output_types = self.mapping_output_combo.checkedItems()
        if not output_types:
            output_types = [MAPPING_OUTPUT_TYPES[0]]
        meshes = []
        if self.mapping_mesh_phase_checkbox.isChecked():
            meshes.append("Phase")
        if self.mapping_mesh_mod_checkbox.isChecked():
            meshes.append("Modulation")
        phase_range, mod_range = self._resolve_mesh_ranges()
        return {
            "output_types": output_types,
            "frequency": float(self.mapping_frequency_spin.text() or 0.0),
            "harmonic": self.mapping_harmonic_spin.value(),
            "colormap": self.mapping_colormap_combo.currentText(),
            "contrast_limits": self._contrast_value(self.mapping_contrast),
            "color_by": self.mapping_color_by_combo.currentText(),
            "meshes": meshes,
            "mesh_colormap": self.mapping_mesh_colormap_combo.currentText(),
            "mesh_alpha": self.mapping_mesh_alpha_spin.value(),
            "mesh_phase_range": phase_range,
            "mesh_modulation_range": mod_range,
            "mesh_clip_semicircle": (
                self.mapping_mesh_clip_checkbox.isChecked()
            ),
        }

    def _collect_fret(self):
        """Return the FRET tab's settings as a dict for the run."""
        return {
            "donor_lifetime": self.fret_donor_lifetime_spin.value(),
            "frequency": float(self.fret_frequency_spin.text() or 0.0),
            "harmonic": self.fret_harmonic_spin.value(),
            "donor_background": self.fret_background_spin.value(),
            "donor_fretting": self.fret_fretting_spin.value(),
            "background_real": self.fret_bg_real_spin.value(),
            "background_imag": self.fret_bg_imag_spin.value(),
            "colormap": self.fret_colormap_combo.currentText(),
            "contrast_limits": self._contrast_value(self.fret_contrast),
        }

    def _collect_selection(self):
        """Return the Selection tab's settings as a dict for the run."""
        cursors = []
        for entry in self._cursor_rows:
            cursor_type = entry["type"].currentData()
            color = entry["color"].color().name()
            if cursor_type == "circular":
                cursors.append(
                    {
                        "type": "circular",
                        "g": float(entry["g"].text() or 0.0),
                        "s": float(entry["s"].text() or 0.0),
                        "radius": entry["radius"].value(),
                        "color": color,
                    }
                )
            elif cursor_type == "elliptic":
                cursors.append(
                    {
                        "type": "elliptic",
                        "g": float(entry["g"].text() or 0.0),
                        "s": float(entry["s"].text() or 0.0),
                        "radius": entry["radius"].value(),
                        "radius_minor": entry["radius_minor"].value(),
                        "angle": float(np.deg2rad(entry["angle"].value())),
                        "color": color,
                    }
                )
            else:
                cursors.append(
                    {
                        "type": "polar",
                        "phase_min": entry["phase_min"].value(),
                        "phase_max": entry["phase_max"].value(),
                        "modulation_min": entry["mod_min"].value(),
                        "modulation_max": entry["mod_max"].value(),
                        "color": color,
                    }
                )
        return {
            "harmonic": self.selection_harmonic_spin.value(),
            "mode": self.selection_mode_combo.currentData(),
            "cursors": cursors,
            "cluster": {
                "clusters": self.cluster_count_spin.value(),
                "sigma": self.cluster_sigma_spin.value(),
            },
        }

    def _collect_filter_kwargs(self):
        """Return the filter keyword arguments for the chosen filter method."""
        kwargs = {}
        method = self.filter_method_combo.currentText()
        if method == "Median":
            kwargs["filter_method"] = "median"
            kwargs["size"] = self.median_size_spin.value()
            kwargs["repeat"] = self.median_repeat_spin.value()
        elif method == "Wavelet (binlet pawFLIM)":
            kwargs["filter_method"] = "wavelet"
            kwargs["sigma"] = self.wavelet_sigma_spin.value()
            kwargs["levels"] = self.wavelet_levels_spin.value()

        threshold_method = self.threshold_method_combo.currentText()
        if threshold_method != "None":
            kwargs["threshold_method"] = threshold_method.lower()
            if threshold_method == "Manual":
                kwargs["threshold"] = self.threshold_min_spin.value()
                if self.threshold_max_spin.value() > 0:
                    kwargs["threshold_upper"] = self.threshold_max_spin.value()
        return kwargs

    def _collect_components(self, harmonics):
        """Return the component settings for *harmonics* as a dict for the run."""
        self._sync_component_coords()

        names, colormaps = [], []
        for index, entry in enumerate(self._component_rows, start=1):
            names.append(entry["name"].text().strip() or f"Component {index}")
            colormaps.append(entry["colormap"].currentText())

        count = len(self._component_rows)
        if (
            count == 2
            and self.analysis_type_combo.currentText() == "Linear Projection"
        ):
            analysis_type = "linear"
        else:
            analysis_type = "fit"

        contrast = self._contrast_value(self.components_contrast)
        config = {
            "analysis_type": analysis_type,
            "names": names,
            "colormaps": colormaps,
            "contrast_limits": contrast,
            "line_style": dict(self._component_line_style),
            "label_style": dict(self._component_label_style),
            "fractions_colormap": _colormap_color_list(
                colormaps[0] if colormaps else "jet"
            ),
            "colormap_contrast_limits": contrast or (0.0, 1.0),
        }

        required = required_component_harmonics(count)
        if required <= 1:
            harmonic = self._component_current_harmonic
            reals, imags = self._component_coords_at(harmonic)
            config.update(
                {
                    "component_real": reals,
                    "component_imag": imags,
                    "harmonic": harmonic,
                }
            )
            return config

        active = self._active_component_harmonics()
        if len(active) < required:
            raise ValueError(
                f"Component Fit with {count} components requires component "
                f"locations at {required} harmonics, but only "
                f"{len(active) or 'none'} "
                f"({active if active else 'no harmonics'}) have a location for "
                "every component. Use the 'Harmonic' selector in the "
                "Components tab to set G/S at each required harmonic."
            )
        used = active[:required]
        reals_2d, imags_2d = [], []
        for harmonic in used:
            reals, imags = self._component_coords_at(harmonic)
            reals_2d.append(reals)
            imags_2d.append(imags)
        config.update(
            {
                "component_real": reals_2d,
                "component_imag": imags_2d,
                "harmonics": used,
                "harmonic": used[0],
            }
        )
        return config

    def _component_coords_at(self, harmonic):
        """Return ``(reals, imags)`` of every component at ``harmonic``."""
        reals, imags = [], []
        for entry in self._component_rows:
            g, s = entry["coords"].get(harmonic, (0.0, 0.0))
            reals.append(float(g))
            imags.append(float(s))
        return reals, imags

    def _resolve_calibration_map(self, harmonics):
        """Return a per-subfolder calibration map, or ``None`` if disabled.

        The returned dict maps a subfolder key to
        ``{"phi_zero", "mod_zero"}``. A ``"*"`` key means the same calibration
        applies to every file.
        """
        if not self.calibration_group.isChecked():
            return None

        source = self.calib_source_combo.currentData()
        if source == "copied":
            if not self._copied_calibration:
                raise ValueError(
                    "Calibration source is 'copied' but no calibration has "
                    "been copied. Use 'Copy settings…' first."
                )
            return {"*": self._copied_calibration}

        frequency = float(self.calib_frequency_spin.text() or 0.0)
        lifetime = self.calib_lifetime_spin.value()

        if source == "same":
            reference = self._resolve_same_reference(harmonics)
            phi_zero, mod_zero = compute_calibration_parameters(
                reference, frequency, lifetime
            )
            return {
                "*": {
                    "phi_zero": phi_zero,
                    "mod_zero": mod_zero,
                    "harmonics": reference.metadata.get("harmonics"),
                }
            }

        result = {}
        for key, edit in self._subfolder_ref_edits.items():
            path = edit.text().strip()
            if not path:
                raise ValueError(
                    "No calibration reference selected for subfolder "
                    f"'{key or '(root)'}'."
                )
            reference = self._load_reference_layer(path, harmonics)
            phi_zero, mod_zero = compute_calibration_parameters(
                reference, frequency, lifetime
            )
            result[key] = {
                "phi_zero": phi_zero,
                "mod_zero": mod_zero,
                "harmonics": reference.metadata.get("harmonics"),
            }
        if not result:
            raise ValueError(
                "No subfolders found to assign references. Scan a folder and "
                "select a format first."
            )
        return result

    def _resolve_same_reference(self, harmonics):
        """Return the single reference used for every file, for *harmonics*."""
        file_path = self.calib_file_edit.text().strip()
        if file_path:
            return self._load_reference_layer(file_path, harmonics)
        name = self.calib_reference_combo.currentText()
        if name and name in self.viewer.layers:
            return self.viewer.layers[name]
        raise ValueError(
            "Calibration is enabled but no reference layer or file was "
            "selected."
        )

    def run_batch(self):
        """Run the configured pipeline over every file of the chosen format."""
        ext = self.format_combobox.currentData()
        if not ext or ext not in self._scanned:
            show_error("Select an input folder with supported files.")
            return
        files = self._scanned[ext]
        if not files:
            show_error("No files of the selected format were found.")
            return
        if not self._export_folder:
            show_error("Select an export folder.")
            return

        output_types = []
        if self.export_ometiff_checkbox.isChecked():
            output_types.append("ome.tif")
        if self.export_csv_checkbox.isChecked():
            output_types.append("csv")
        if self.export_image_checkbox.isChecked():
            output_types.append("png")
        if not output_types and not self._has_extra_outputs():
            show_error(
                "Select an export format, or enable a phasor plot or a "
                "per-tab export."
            )
            return

        try:
            harmonics = parse_harmonics(self.harmonics_edit.text())
        except ValueError:
            show_error("Invalid harmonics value.")
            return

        reader_options = self.read_options_widget.get_reader_options()

        try:
            pipeline = self.build_pipeline(harmonics)
            calibration_map = self._resolve_calibration_map(harmonics)
        except ValueError as exc:
            show_error(str(exc))
            return

        want_individual = self.plot_individual_checkbox.isChecked()
        want_combined = self.plot_combined_checkbox.isChecked()

        display = self._collect_plot_settings("individual")
        display_combined = self._collect_plot_settings("combined")
        persist_plot = want_individual or want_combined
        plot_jobs = self._collect_plot_jobs(pipeline)
        export_centers = self.plot_centers_checkbox.isChecked()
        center_rows = []

        analysis_jobs = self._collect_analysis_export_jobs(pipeline)
        analysis_stats = {job["name"]: [] for job in analysis_jobs}

        signal_formats = self.signal_format_combo.checkedItems()
        signal_enabled = (
            self._signal_available
            and self.signal_group.isChecked()
            and bool(signal_formats)
        )
        want_signal_individual = (
            signal_enabled and self.signal_individual_checkbox.isChecked()
        )
        want_signal_combined = (
            signal_enabled and self.signal_combined_checkbox.isChecked()
        )
        if want_signal_individual or want_signal_combined:
            self._signal_export_cfg = {
                "individual": want_signal_individual,
                "combined": want_signal_combined,
                "png": "PNG" in signal_formats,
                "csv": "CSV" in signal_formats,
                "normalize": self.signal_normalize_combo.currentData(),
                "channel_mode": self.signal_channel_combo.currentData(),
                "color": self.signal_color.color().name(),
                "white_background": self.plot_white_bg_checkbox.isChecked(),
                "legend": self.plot_legend_checkbox.isChecked(),
            }
            self._signal_combined = {}
            processed_exts = set(extension_mapping["processed"])
            raw_exts = set(extension_mapping["raw"])
            if ext not in processed_exts or ext in raw_exts:
                reader_options = {**reader_options, "_keep_signal": True}
        else:
            self._signal_export_cfg = None

        self._selection_stats_config = (
            pipeline.selection
            if (
                pipeline.selection is not None
                and self.selection_stats_checkbox.isChecked()
            )
            else None
        )
        self._selection_stats_rows = []

        streaming = self.streaming_checkbox.isChecked()
        spill_dir = (
            tempfile.mkdtemp(prefix="napari_phasors_batch_")
            if streaming
            else None
        )
        combined_phasor = want_combined

        tab_combined = want_combined
        aggregate = {
            "streaming": streaming,
            "combined_phasor": combined_phasor,
            "plot_type": display_combined.get("plot_type", "Histogram"),
            "contour": combined_phasor,
            "centers": want_combined and display_combined.get("show_center"),
            "contour_data": {},
            "centers_data": {},
            "hist_data": {},
            "group_meta": {},
            "tab_phasor": tab_combined,
            "tab_phasor_data": {},  # suffix -> {(harmonic, key): [(r, i), ...]}
            "tab_phasor_overlay": {},  # suffix -> overlay spec
            "tab_phasor_subfolder": {},  # suffix -> subfolder name
        }
        if streaming:
            aggregate["contour_store_r"] = _SpillStore(
                os.path.join(spill_dir, "contour_r")
            )
            aggregate["contour_store_i"] = _SpillStore(
                os.path.join(spill_dir, "contour_i")
            )
            aggregate["contour_struct"] = {}
            aggregate["hist_store"] = _SpillStore(
                os.path.join(spill_dir, "hist")
            )
            aggregate["hist_struct"] = {}
            aggregate["tab_phasor_store_r"] = _SpillStore(
                os.path.join(spill_dir, "tab_phasor_r")
            )
            aggregate["tab_phasor_store_i"] = _SpillStore(
                os.path.join(spill_dir, "tab_phasor_i")
            )
            aggregate["tab_phasor_struct"] = (
                {}
            )  # suffix -> set((harmonic, key))

        suffix = self.suffix_edit.text()
        preserve = self.preserve_paths_checkbox.isChecked()
        load_into_viewer = self.load_into_viewer_checkbox.isChecked()
        workers = self.threads_spin.value()
        masks_enabled = self.masks_group.isChecked()
        self._auto_tabs = self._auto_contrast_tabs()
        self._global_contrast = {}
        self._global_contrast_acc = {}
        self._deferred_exports = []
        self._deferred_dir = (
            tempfile.mkdtemp(prefix="napari_phasors_defer_")
            if self._auto_tabs
            else None
        )
        self._deferred_store = (
            _SpillStore(os.path.join(self._deferred_dir, "arrays"))
            if self._deferred_dir
            else None
        )

        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(len(files))
        self.progress_bar.setValue(0)

        def read_compute(path):
            calibration = self._calibration_for(path, calibration_map)
            mask_spec = (
                self._mask_for(path) if masks_enabled else (None, False)
            )
            return self._read_compute_file(
                path,
                ext,
                reader_options,
                harmonics,
                pipeline,
                calibration,
                mask_spec,
            )

        def emit(path, results):
            for layer, extra_layers in results:
                self._emit_file_outputs(
                    path,
                    layer,
                    extra_layers,
                    ext,
                    output_types,
                    suffix,
                    preserve,
                    load_into_viewer,
                    display,
                    persist_plot,
                    plot_jobs,
                    export_centers,
                    center_rows,
                    analysis_jobs,
                    analysis_stats,
                    aggregate,
                )

            if self._signal_export_cfg is not None:
                self._emit_signal_outputs(path, results, ext, suffix, preserve)

        processed = 0
        failed = []

        def update_progress(index):
            self.progress_bar.setValue(index + 1)
            self.status_label.setText(
                f"Processed {index + 1}/{len(files)} files…"
            )
            QApplication.processEvents()

        if workers > 1 and len(files) > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    (path, executor.submit(read_compute, path))
                    for path in files
                ]
                for index, (path, future) in enumerate(futures):
                    try:
                        emit(path, future.result())
                        processed += 1
                    except Exception as exc:  # noqa: BLE001
                        failed.append((os.path.basename(path), str(exc)))
                    update_progress(index)
        else:
            for index, path in enumerate(files):
                try:
                    emit(path, read_compute(path))
                    processed += 1
                except Exception as exc:  # noqa: BLE001
                    failed.append((os.path.basename(path), str(exc)))
                update_progress(index)

        try:
            self._flush_deferred_exports()
            if export_centers and center_rows:
                self._write_centers_csv(center_rows)
            self._write_analysis_export_csvs(analysis_jobs, analysis_stats)
            self._write_selection_stats_csv()
            self._write_aggregate_outputs(aggregate, display_combined)
            self._write_tab_phasor_outputs(aggregate, display_combined)
            if (
                self._signal_export_cfg is not None
                and self._signal_export_cfg["combined"]
            ):
                self._write_signal_combined()
        finally:
            if spill_dir:
                shutil.rmtree(spill_dir, ignore_errors=True)
            if self._deferred_dir:
                shutil.rmtree(self._deferred_dir, ignore_errors=True)

        self.progress_bar.setVisible(False)
        summary = f"Batch complete: {processed}/{len(files)} files processed."
        if failed:
            summary += f" {len(failed)} failed."
            details = "\n".join(f"  • {name}: {msg}" for name, msg in failed)
            show_error(f"{summary}\n{details}")
        else:
            show_info(summary)
        self.status_label.setText(summary)

    def _calibration_for(self, path, calibration_map):
        """Resolve the calibration params for ``path`` from the map."""
        if calibration_map is None:
            return None
        if "*" in calibration_map:
            return calibration_map["*"]
        key = self._subfolder_key(path)
        params = calibration_map.get(key)
        if params is None:
            raise ValueError(
                f"No calibration reference for subfolder '{key or '(root)'}'."
            )
        return params

    def _read_compute_file(
        self,
        path,
        ext,
        reader_options,
        harmonics,
        pipeline,
        calibration,
        mask_spec=(None, False),
    ):
        """Read ``path`` and apply the pipeline (thread-safe, no IO/render).

        Returns a list of ``(layer, extra_layers)``."""
        import dataclasses

        reader = napari_get_reader(
            path, reader_options=reader_options, harmonics=harmonics
        )
        if reader is None:
            raise ValueError("no reader")

        mask_path, invert = mask_spec
        mask = None
        if mask_path:
            mask = {"array": _load_mask_array(mask_path), "invert": invert}

        local_pipeline = dataclasses.replace(
            pipeline, calibration=calibration, mask=mask
        )
        results = []
        for layer_data in reader(path):
            data = layer_data[0]
            add_kw = dict(layer_data[1])
            layer = Image(
                data,
                name=add_kw.get("name", os.path.basename(path)),
                metadata=add_kw.get("metadata", {}),
            )
            extra_layers = apply_pipeline(layer, local_pipeline)
            if self._signal_export_cfg is not None:
                self._attach_signal_profile(layer, mask)
            results.append((layer, extra_layers))
        return results

    def _attach_signal_profile(self, layer, mask):
        """Store the file's 1-D signal profile in ``layer.metadata``."""
        meta = layer.metadata
        full = meta.pop("signal_full", None)
        axis = meta.pop("signal_axis", None)
        profile = None
        if full is not None:
            profile = _masked_signal_mean(full, axis, mask)
        else:
            sig = meta.get("summed_signal")
            if sig is None:
                sig = meta.get("settings", {}).get("summed_signal")
            if sig is not None:
                arr = np.asarray(sig, dtype=float).ravel()
                mean_img = meta.get("original_mean")
                n = (
                    int(np.asarray(mean_img).size)
                    if mean_img is not None
                    else 0
                )
                profile = arr / n if n > 0 else arr
        if profile is not None:
            meta["_signal_profile"] = np.asarray(profile, dtype=float).ravel()

    def _auto_contrast_tabs(self):
        """Return the set of analysis-tab keys with contrast set to Auto."""
        tabs = set()
        if (
            self.components_group.isChecked()
            and self.components_contrast["auto"].isChecked()
        ):
            tabs.add("components")
        if (
            self.mapping_group.isChecked()
            and self.mapping_contrast["auto"].isChecked()
        ):
            tabs.add("phasor_mapping")
        if (
            self.fret_group.isChecked()
            and self.fret_contrast["auto"].isChecked()
        ):
            tabs.add("fret")
        return tabs

    def _subfolder_for_layer(self, layer_name):
        """Return the analysis-tab *key* that produced ``layer_name``."""
        if "fraction: " in layer_name:
            return "components"
        elif any(
            layer_name.startswith(f"{t}: ")
            for t in [
                "Phase",
                "Modulation",
                "Normal Lifetime",
                "Apparent Phase Lifetime",
                "Apparent Modulation Lifetime",
            ]
        ):
            return "phasor_mapping"
        elif layer_name.startswith("FRET efficiency: "):
            return "fret"
        elif layer_name.startswith(
            ("Cursor selection: ", "Cluster selection: ")
        ):
            return "selection"
        return None

    def _clean_layer_name(self, layer_name):
        """Return *layer_name* without its trailing ``": <source>"`` suffix."""
        if ": " in layer_name:
            return layer_name.split(": ", 1)[0]
        return layer_name

    def _emit_file_outputs(
        self,
        path,
        layer,
        extra_layers,
        ext,
        output_types,
        suffix,
        preserve,
        load_into_viewer,
        display=None,
        persist_plot=False,
        plot_jobs=None,
        export_centers=False,
        center_rows=None,
        analysis_jobs=None,
        analysis_stats=None,
        aggregate=None,
    ):
        """Export files, render plots and accumulate (main-thread only)."""
        group_key, group_name, group_color = self._group_for(
            os.path.basename(path)
        )

        if persist_plot and display is not None:
            _store_plot_settings(layer, display, self._group_config)

        non_png_types = [t for t in output_types if t != "png"]
        if non_png_types:
            self._export_layer_files(
                layer, path, ext, suffix, preserve, non_png_types
            )
        if "png" in output_types:
            self._export_layer_files(
                layer,
                path,
                ext,
                suffix,
                preserve,
                ["png"],
                name_suffix="_Intensity_Image",
            )

        job_by_tab = {j["name"]: j for j in (analysis_jobs or [])}
        for extra in extra_layers:
            self._handle_output_layer(
                extra,
                layer,
                path,
                ext,
                suffix,
                preserve,
                output_types,
                job_by_tab,
                analysis_stats,
                aggregate,
                (group_key, group_name, group_color),
            )

        self._record_selection_stats(layer, extra_layers)

        if export_centers and center_rows is not None:
            self._record_phasor_centers(layer, center_rows)

        if aggregate is not None:
            self._accumulate_phasor_aggregate(
                layer, aggregate, (group_key, group_name, group_color)
            )

        want_individual = self.plot_individual_checkbox.isChecked()
        want_combined = self.plot_combined_checkbox.isChecked()
        for job in plot_jobs or []:
            tab = job.get("tab")
            if tab is None:
                subfolder = os.path.join(
                    "Individual image analysis", "Phasor Plots"
                )
            else:
                subfolder = os.path.join(
                    self._TAB_FOLDERS[tab],
                    "Individual image analysis",
                    "Phasor Plots",
                )
            if want_individual:
                job_base_out = self._derive_output_path(
                    path, ext, "", preserve, subfolder=subfolder
                )
                self._render_phasor_plot(
                    layer, job_base_out, display, job["suffix"], job["overlay"]
                )
            if (
                want_combined
                and aggregate is not None
                and job.get("tab") is not None
            ):
                self._accumulate_tab_phasor(
                    layer,
                    aggregate,
                    job,
                    (group_key, group_name, group_color),
                )

        if load_into_viewer:
            self.viewer.add_image(
                layer.data, name=layer.name, metadata=layer.metadata
            )
            for extra in extra_layers:
                self.viewer.add_image(extra.data, name=extra.name)

    def _derive_output_path(
        self, src_path, ext, suffix, preserve, subfolder=None
    ):
        """Return the output path *without* extension for ``src_path``."""
        filename = os.path.basename(src_path)
        stem = (
            filename[: -len(ext)]
            if filename.lower().endswith(ext)
            else (os.path.splitext(filename)[0])
        )
        if preserve and self._input_folder:
            rel_dir = os.path.relpath(
                os.path.dirname(src_path), self._input_folder
            )
        else:
            rel_dir = ""

        target_dir = self._export_folder
        if subfolder:
            target_dir = os.path.join(target_dir, subfolder)
        if rel_dir:
            target_dir = os.path.join(target_dir, rel_dir)

        target_dir = os.path.normpath(target_dir)
        os.makedirs(target_dir, exist_ok=True)
        return os.path.join(target_dir, f"{stem}{suffix}")

    def _layer_output_subfolder(self, out_type, tab):
        """Return the subfolder for a layer file of ``out_type``."""
        if tab is None:
            if out_type == "png":
                return "Intensity Images"
            return self._TYPE_DIRS[out_type]
        return os.path.join(
            self._TAB_FOLDERS[tab],
            "Individual image analysis",
            self._TYPE_DIRS[out_type],
        )

    def _export_layer_files(
        self,
        layer,
        src_path,
        ext,
        suffix,
        preserve,
        output_types,
        *,
        tab=None,
        name_suffix="",
        include_colorbar=False,
    ):
        """Write ``layer`` for each requested type into its typed subfolder."""
        for out_type in output_types:
            subfolder = self._layer_output_subfolder(out_type, tab)
            base = self._derive_output_path(
                src_path, ext, suffix, preserve, subfolder=subfolder
            )
            target = f"{base}{name_suffix}.{out_type}"
            if out_type == "ome.tif":
                write_ome_tiff(target, layer)
            elif out_type == "csv":
                export_layer_as_csv(target, layer)
            elif out_type == "png":
                export_layer_as_image(
                    target,
                    layer,
                    include_colorbar=include_colorbar,
                    dpi=self._export_dpi(),
                )

    @staticmethod
    def _checked_formats(combo):
        """Return selected export formats from a format combobox as types."""
        mapping = {"PNG": "png", "CSV": "csv"}
        return [
            mapping[name] for name in combo.checkedItems() if name in mapping
        ]

    def _collect_analysis_export_jobs(self, pipeline):
        """Build per-tab export jobs (image / histogram / stats) for outputs."""
        jobs = []
        specs = [
            (
                "components",
                pipeline.components,
                self.components_export_controls,
            ),
            ("phasor_mapping", pipeline.mapping, self.mapping_export_controls),
            ("fret", pipeline.fret, self.fret_export_controls),
        ]
        for name, config, controls in specs:
            if config is None:
                continue
            stats = controls["stats"].isChecked()
            image_formats = self._checked_formats(controls["image"])
            histogram_formats = self._checked_formats(controls["histogram"])
            if not (stats or image_formats or histogram_formats):
                continue
            jobs.append(
                {
                    "name": name,
                    "config": config,
                    "stats": stats,
                    "histogram": bool(histogram_formats),
                    "image_formats": image_formats,
                    "histogram_formats": histogram_formats,
                }
            )

        if pipeline.selection is not None:
            stats = self.selection_stats_checkbox.isChecked()
            image_formats = self._checked_formats(self.selection_export_combo)
            if stats or image_formats:
                jobs.append(
                    {
                        "name": "selection",
                        "config": pipeline.selection,
                        "stats": stats,
                        "histogram": False,
                        "image_formats": image_formats,
                        "histogram_formats": [],
                    }
                )

        return jobs

    def _handle_output_layer(
        self,
        output,
        layer,
        src_path,
        ext,
        suffix,
        preserve,
        output_types,
        job_by_tab,
        stats_accum,
        aggregate,
        group,
    ):
        """Export + record one analysis output layer from the single pass."""
        tab = self._subfolder_for_layer(output.name)
        label = self._clean_layer_name(output.name)
        safe_label = _safe_suffix(label)
        is_auto = tab in self._auto_tabs
        job = job_by_tab.get(tab)

        values = np.asarray(output.data).ravel()
        valid = values[np.isfinite(values)]

        if tab in self._TAB_IMAGE_TABS:
            image_formats = list(job["image_formats"]) if job else []
        else:
            image_formats = list(output_types)

        immediate_image = [
            t for t in image_formats if not (is_auto and t == "png")
        ]
        if immediate_image:
            self._export_layer_files(
                output,
                src_path,
                ext,
                suffix,
                preserve,
                immediate_image,
                tab=tab,
                name_suffix=f"_{safe_label}",
                include_colorbar=self.export_colorbar_checkbox.isChecked(),
            )

        if is_auto and valid.size:
            low, high = float(np.nanmin(valid)), float(np.nanmax(valid))
            current = self._global_contrast_acc.get(label)
            if current is None:
                self._global_contrast_acc[label] = [low, high]
            else:
                current[0] = min(current[0], low)
                current[1] = max(current[1], high)

        if job and job["stats"] and stats_accum is not None:
            row = _compute_stats(valid, 100)
            row["file"] = layer.name
            row["output"] = label
            stats_accum[tab].append(row)

        cmap_colors = (
            getattr(output.colormap, "colors", None)
            if hasattr(output, "colormap")
            else None
        )
        hist_formats = job["histogram_formats"] if job else []
        want_hist = bool(hist_formats and valid.size)
        want_image_png = is_auto and "png" in image_formats

        if is_auto:
            if want_image_png or want_hist:
                self._defer_output_export(
                    output,
                    src_path,
                    ext,
                    suffix,
                    preserve,
                    tab,
                    label,
                    safe_label,
                    cmap_colors,
                    hist_formats if want_hist else [],
                    want_image_png,
                )
        elif want_hist:
            self._write_histogram_outputs(
                src_path,
                ext,
                suffix,
                preserve,
                tab,
                label,
                safe_label,
                valid,
                hist_formats,
                cmap_colors,
                getattr(output, "contrast_limits", None),
            )

        if (
            job
            and (job["stats"] or job["histogram"])
            and aggregate is not None
            and group is not None
        ):
            key, gname, gcolor = group
            aggregate["group_meta"][key] = (gname, gcolor)
            if aggregate.get("streaming"):
                members = (
                    aggregate["hist_struct"]
                    .setdefault(tab, {})
                    .setdefault(label, {})
                    .setdefault(key, [])
                )
                member_key = (tab, label, key, len(members))
                aggregate["hist_store"].append(member_key, valid)
                members.append(member_key)
            else:
                tab_data = aggregate["hist_data"].setdefault(tab, {})
                label_data = tab_data.setdefault(label, {})
                label_data.setdefault(key, []).append(valid)

    def _write_histogram_outputs(
        self,
        src_path,
        ext,
        suffix,
        preserve,
        tab,
        label,
        safe_label,
        valid,
        hist_formats,
        cmap_colors,
        value_range,
    ):
        """Write the per-file histogram in the requested formats (PNG/CSV)."""
        if "png" in hist_formats:
            png_base = self._derive_output_path(
                src_path,
                ext,
                suffix,
                preserve,
                subfolder=os.path.join(
                    self._TAB_FOLDERS[tab],
                    "Individual image analysis",
                    "Histograms",
                    "PNG",
                ),
            )
            _save_histogram_png(
                valid,
                100,
                f"{png_base}_{safe_label}_histogram.png",
                label,
                colormap_colors=cmap_colors,
                contrast_limits=value_range,
                value_range=value_range,
                dpi=self._export_dpi(),
            )
        if "csv" in hist_formats:
            csv_base = self._derive_output_path(
                src_path,
                ext,
                suffix,
                preserve,
                subfolder=os.path.join(
                    self._TAB_FOLDERS[tab],
                    "Individual image analysis",
                    "Histograms",
                    "CSV",
                ),
            )
            _save_histogram_csv(
                valid,
                100,
                f"{csv_base}_{safe_label}_histogram.csv",
                value_range=value_range,
            )

    def _defer_output_export(
        self,
        output,
        src_path,
        ext,
        suffix,
        preserve,
        tab,
        label,
        safe_label,
        cmap_colors,
        hist_formats,
        want_image_png,
    ):
        """Spill an output array and queue its range-dependent files."""
        array = np.asarray(output.data)
        store_key = len(self._deferred_exports)
        self._deferred_store.append(store_key, array)
        self._deferred_exports.append(
            {
                "key": store_key,
                "shape": tuple(array.shape),
                "src_path": src_path,
                "ext": ext,
                "suffix": suffix,
                "preserve": preserve,
                "tab": tab,
                "label": label,
                "safe_label": safe_label,
                "cmap_colors": cmap_colors,
                "colormap": getattr(output, "colormap", None),
                "hist_formats": list(hist_formats),
                "image_png": want_image_png,
            }
        )

    def _flush_deferred_exports(self):
        """Write deferred Auto outputs using the pooled all-files range."""
        self._global_contrast = {
            label: (lo, hi)
            for label, (lo, hi) in self._global_contrast_acc.items()
            if hi > lo
        }
        for rec in self._deferred_exports:
            array = self._deferred_store.load(rec["key"]).reshape(rec["shape"])
            clim = self._global_contrast.get(rec["label"])
            valid = array.ravel()
            valid = valid[np.isfinite(valid)]
            if rec["hist_formats"] and valid.size:
                self._write_histogram_outputs(
                    rec["src_path"],
                    rec["ext"],
                    rec["suffix"],
                    rec["preserve"],
                    rec["tab"],
                    rec["label"],
                    rec["safe_label"],
                    valid,
                    rec["hist_formats"],
                    rec["cmap_colors"],
                    clim,
                )
            if rec["image_png"]:
                image = Image(array, name=rec["label"])
                if rec["colormap"] is not None:
                    image.colormap = rec["colormap"]
                if clim is not None:
                    image.contrast_limits_range = [clim[0], clim[1]]
                    image.contrast_limits = [clim[0], clim[1]]
                base = self._derive_output_path(
                    rec["src_path"],
                    rec["ext"],
                    rec["suffix"],
                    rec["preserve"],
                    subfolder=os.path.join(
                        self._TAB_FOLDERS[rec["tab"]],
                        "Individual image analysis",
                        "Images",
                    ),
                )
                export_layer_as_image(
                    f"{base}_{rec['safe_label']}.png",
                    image,
                    include_colorbar=self.export_colorbar_checkbox.isChecked(),
                    dpi=self._export_dpi(),
                )

    def _write_analysis_export_csvs(self, jobs, stats_accum):
        """Write each job's accumulated statistics to its export CSV."""
        for job in jobs:
            name = job["name"]
            if job["stats"] and stats_accum.get(name):
                tab_name = self._TAB_FOLDERS.get(name, "")
                target_dir = (
                    os.path.join(
                        self._export_folder,
                        tab_name,
                        "Combined analysis",
                        "Statistics",
                    )
                    if tab_name
                    else os.path.join(
                        self._export_folder,
                        "Combined analysis",
                        "Statistics",
                    )
                )
                os.makedirs(target_dir, exist_ok=True)
                path = os.path.join(target_dir, f"{name}_statistics.csv")
                with open(path, "w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(
                        [
                            "File",
                            "Output",
                            "Mean",
                            "Median",
                            "Std Dev",
                            "Center of Mass",
                            "N pixels",
                        ]
                    )
                    for row in stats_accum[name]:
                        writer.writerow(
                            [
                                row["file"],
                                row["output"],
                                f"{row['mean']:.6f}",
                                f"{row['median']:.6f}",
                                f"{row['std']:.6f}",
                                f"{row['com']:.6f}",
                                row["n"],
                            ]
                        )

    # -- Selection statistics ----------------------------------------------

    def _record_selection_stats(self, layer, extra_layers):
        """Accumulate per-region selection statistics for ``layer``."""
        selection = getattr(self, "_selection_stats_config", None)
        if selection is None:
            return
        selection_map = next(
            (
                np.asarray(extra.data)
                for extra in extra_layers
                if isinstance(extra, Labels)
            ),
            None,
        )
        if selection_map is None:
            return
        for row in _selection_statistics(layer, selection, selection_map):
            row["file"] = layer.name
            self._selection_stats_rows.append(row)

    def _write_selection_stats_csv(self):
        """Write the accumulated selection statistics to a single CSV."""
        rows = getattr(self, "_selection_stats_rows", None)
        if not rows:
            return
        target_dir = os.path.join(
            self._export_folder, "Combined analysis", "Statistics"
        )
        os.makedirs(target_dir, exist_ok=True)
        path = os.path.join(target_dir, "selection_statistics.csv")

        def _fmt(value):
            return f"{value:.6f}" if isinstance(value, float) else value

        with open(path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "File",
                    "Region",
                    "Type",
                    "G",
                    "S",
                    "Radius",
                    "Pixel Count",
                    "% of Valid Pixels",
                ]
            )
            for row in rows:
                writer.writerow(
                    [
                        row["file"],
                        row["region"],
                        row["type"],
                        _fmt(row["g"]),
                        _fmt(row["s"]),
                        _fmt(row["radius"]),
                        row["count"],
                        f"{row['percent']:.2f}",
                    ]
                )

    def _record_phasor_centers(self, layer, center_rows):
        """Append the per-harmonic phasor center of ``layer``."""
        mean = layer.metadata.get("original_mean")
        if mean is None:
            return
        harmonics = np.atleast_1d(layer.metadata.get("harmonics"))
        for harmonic in harmonics:
            real, imag = _select_harmonic_arrays(layer, int(harmonic))
            if real is None:
                continue
            _, center_real, center_imag = phasor_center(mean, real, imag)
            center_rows.append(
                {
                    "file": layer.name,
                    "harmonic": int(harmonic),
                    "center_real": float(center_real),
                    "center_imag": float(center_imag),
                }
            )

    def _write_centers_csv(self, center_rows):
        """Write the collected phasor centers to ``phasor_centers.csv``."""
        target_dir = os.path.join(self._export_folder, "Phasor Centers")
        os.makedirs(target_dir, exist_ok=True)
        path = os.path.join(target_dir, "phasor_centers.csv")
        with open(path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["File", "Harmonic", "Center G", "Center S"])
            for row in center_rows:
                writer.writerow(
                    [
                        row["file"],
                        row["harmonic"],
                        f"{row['center_real']:.6f}",
                        f"{row['center_imag']:.6f}",
                    ]
                )

    def _accumulate_phasor_aggregate(self, layer, aggregate, group):
        """Accumulate per-group phasor data for combined plots."""
        if not (aggregate["contour"] or aggregate["centers"]):
            return
        key, group_name, group_color = group
        aggregate["group_meta"][key] = (group_name, group_color)
        harmonics = np.atleast_1d(layer.metadata.get("harmonics"))
        mean = layer.metadata.get("original_mean")
        for harmonic in harmonics:
            real, imag = _select_harmonic_arrays(layer, int(harmonic))
            if real is None:
                continue
            if aggregate["centers"] and mean is not None:
                flat_mean = np.asarray(mean).ravel()
                flat_r = np.asarray(real).ravel()
                flat_i = np.asarray(imag).ravel()
                m = (
                    np.isfinite(flat_mean)
                    & np.isfinite(flat_r)
                    & np.isfinite(flat_i)
                )
                if m.any():
                    store = aggregate["centers_data"].setdefault(
                        int(harmonic), {}
                    )
                    acc = store.setdefault(key, [0.0, 0.0, 0.0])
                    acc[0] += float(flat_mean[m].sum())
                    acc[1] += float((flat_mean[m] * flat_r[m]).sum())
                    acc[2] += float((flat_mean[m] * flat_i[m]).sum())
            if aggregate["contour"]:
                flat_real = np.asarray(real).ravel()
                flat_imag = np.asarray(imag).ravel()
                finite = np.isfinite(flat_real) & np.isfinite(flat_imag)
                if aggregate.get("streaming"):
                    aggregate["contour_store_r"].append(
                        (int(harmonic), key), flat_real[finite]
                    )
                    aggregate["contour_store_i"].append(
                        (int(harmonic), key), flat_imag[finite]
                    )
                    aggregate["contour_struct"].setdefault(
                        int(harmonic), set()
                    ).add(key)
                else:
                    store = aggregate["contour_data"].setdefault(
                        int(harmonic), {}
                    )
                    store.setdefault(key, []).append(
                        (flat_real[finite], flat_imag[finite])
                    )

    def _accumulate_tab_phasor(self, layer, aggregate, job, group):
        """Pool an analysis-tab phasor plot's coordinates for combined output."""
        if not aggregate.get("tab_phasor"):
            return
        suffix = job["suffix"]
        tab = job.get("tab")
        subfolder = (
            os.path.join(
                self._TAB_FOLDERS[tab],
                "Combined analysis",
                "Phasor Plots",
            )
            if tab in self._TAB_FOLDERS
            else None
        )
        aggregate["tab_phasor_overlay"].setdefault(suffix, job["overlay"])
        aggregate["tab_phasor_subfolder"].setdefault(suffix, subfolder)
        key, group_name, group_color = group
        aggregate["group_meta"][key] = (group_name, group_color)
        harmonics = np.atleast_1d(layer.metadata.get("harmonics"))
        for harmonic in harmonics:
            real, imag = _select_harmonic_arrays(layer, int(harmonic))
            if real is None:
                continue
            flat_real = np.asarray(real).ravel()
            flat_imag = np.asarray(imag).ravel()
            finite = np.isfinite(flat_real) & np.isfinite(flat_imag)
            if aggregate.get("streaming"):
                aggregate["tab_phasor_store_r"].append(
                    (suffix, int(harmonic), key), flat_real[finite]
                )
                aggregate["tab_phasor_store_i"].append(
                    (suffix, int(harmonic), key), flat_imag[finite]
                )
                aggregate["tab_phasor_struct"].setdefault(suffix, set()).add(
                    (int(harmonic), key)
                )
            else:
                store = aggregate["tab_phasor_data"].setdefault(suffix, {})
                store.setdefault((int(harmonic), key), []).append(
                    (flat_real[finite], flat_imag[finite])
                )

    def _write_tab_phasor_outputs(self, aggregate, display):
        """Render combined (merged/grouped) analysis-tab phasor plots."""
        if not aggregate.get("tab_phasor"):
            return
        streaming = aggregate.get("streaming")
        group_meta = aggregate["group_meta"]
        suffixes = (
            aggregate["tab_phasor_struct"]
            if streaming
            else aggregate["tab_phasor_data"]
        )
        for suffix in suffixes:
            overlay = aggregate["tab_phasor_overlay"].get(suffix)
            subfolder = aggregate["tab_phasor_subfolder"].get(suffix)
            target_dir = self._export_folder
            if subfolder:
                target_dir = os.path.join(target_dir, subfolder)
                os.makedirs(target_dir, exist_ok=True)
            entries = self._tab_phasor_items(aggregate, suffix, streaming)

            harmonic_keys = {}
            for harmonic, key in entries:
                harmonic_keys.setdefault(harmonic, []).append(key)
            colors = self._resolve_group_colors(group_meta)
            legend = display.get("show_legend", True)
            is_contour = display.get("plot_type") == "Contour"

            grouped_by_harmonic = {}
            for harmonic, key, real, imag in self._tab_phasor_arrays(
                aggregate, suffix, streaming
            ):
                multi = len(harmonic_keys.get(harmonic, [])) > 1
                if multi:
                    name = group_meta.get(key, (str(key), None))[0]
                    fname = (
                        f"combined_{suffix}_{_safe_suffix(name)}"
                        f"_H{harmonic}.png"
                    )
                    grouped_by_harmonic.setdefault(harmonic, []).append(
                        (key, real, imag)
                    )
                else:
                    fname = f"combined_{suffix}_H{harmonic}.png"
                path = os.path.join(target_dir, fname)
                if is_contour and multi:
                    item = [(key, real, imag)]
                    _save_combined_contour(
                        item,
                        group_meta,
                        colors,
                        display,
                        legend,
                        path,
                        styles=self._contour_key_styles(item),
                        overlay=overlay,
                        dpi=self._export_dpi(),
                    )
                else:
                    _save_phasor_plot_png(
                        real,
                        imag,
                        display,
                        overlay,
                        path,
                        dpi=self._export_dpi(),
                    )
            for harmonic, group_items in grouped_by_harmonic.items():
                all_path = os.path.join(
                    target_dir, f"combined_{suffix}_all_groups_H{harmonic}.png"
                )
                if is_contour:
                    _save_combined_contour(
                        group_items,
                        group_meta,
                        colors,
                        display,
                        legend,
                        all_path,
                        styles=self._contour_key_styles(group_items),
                        overlay=overlay,
                        dpi=self._export_dpi(),
                    )
                else:
                    _save_grouped_overlay_plot(
                        group_items,
                        group_meta,
                        colors,
                        display,
                        legend,
                        all_path,
                        overlay=overlay,
                        dpi=self._export_dpi(),
                    )

    @staticmethod
    def _tab_phasor_items(aggregate, suffix, streaming):
        """Return the list of ``(harmonic, key)`` accumulated for ``suffix``."""
        if streaming:
            return list(aggregate["tab_phasor_struct"].get(suffix, set()))
        return list(aggregate["tab_phasor_data"].get(suffix, {}))

    @staticmethod
    def _tab_phasor_arrays(aggregate, suffix, streaming):
        """Yield ``(harmonic, key, real, imag)`` for ``suffix``, one at a time."""
        if streaming:
            for harmonic, key in aggregate["tab_phasor_struct"].get(
                suffix, set()
            ):
                yield (
                    harmonic,
                    key,
                    aggregate["tab_phasor_store_r"].load(
                        (suffix, harmonic, key)
                    ),
                    aggregate["tab_phasor_store_i"].load(
                        (suffix, harmonic, key)
                    ),
                )
        else:
            for (harmonic, key), arrays in (
                aggregate["tab_phasor_data"].get(suffix, {}).items()
            ):
                yield (
                    harmonic,
                    key,
                    np.concatenate([a[0] for a in arrays]),
                    np.concatenate([a[1] for a in arrays]),
                )

    def _resolve_group_colors(self, group_meta):
        """Return ``{group_key: color}``, filling gaps from the default cycle."""
        colors = {}
        for idx, key in enumerate(group_meta):
            _, color = group_meta[key]
            if not color:
                color = DEFAULT_CURSOR_COLORS[idx % len(DEFAULT_CURSOR_COLORS)]
            colors[key] = color
        return colors

    def _contour_key_styles(self, items):
        """Return ``{key: {mode, colormap, color}}`` for combined contours.

        Maps each accumulated key (single/merged, per-group or per-file) to the
        colormap/solid style chosen in the Contour Layer Settings dialog. Falls
        back to ``None`` (solid color) when no style was configured.
        """
        cfg = self._group_config
        mode = cfg.get("mode", "Merged")
        styles = {}
        for key, _real, _imag in items:
            if mode == "Merged":
                styles[key] = {
                    "mode": cfg.get("contour_merged_style", "colormap"),
                    "colormap": cfg.get("contour_merged_colormap", "jet"),
                    "color": cfg.get("contour_merged_color"),
                }
            elif mode == "Individual layers":
                styles[key] = cfg.get("contour_layer_styles", {}).get(key)
            else:  # Grouped (key is the group id)
                styles[key] = cfg.get("contour_group_styles", {}).get(key)
        return styles

    def _write_aggregate_outputs(self, aggregate, display):
        """Write the combined plots and statistics pooled across every file."""
        group_meta = aggregate["group_meta"]
        colors = self._resolve_group_colors(group_meta)
        legend = display.get("show_legend", True)
        streaming = aggregate.get("streaming")

        plot_type = aggregate.get("plot_type", "Histogram")

        mode = self._group_config.get("mode", "Merged")
        merged_center_color = (
            display.get("center_color") if mode == "Merged" else None
        )
        draw_centers = bool(aggregate.get("centers"))
        if streaming:
            coord_harmonics = sorted(aggregate["contour_struct"])
        else:
            coord_harmonics = sorted(aggregate["contour_data"])
        for harmonic in coord_harmonics:
            items = list(
                self._contour_group_items(aggregate, harmonic, streaming)
            )
            centers = (
                self._group_centers(aggregate, harmonic)
                if draw_centers
                else {}
            )
            combined_dir = os.path.join(
                self._export_folder,
                "Combined analysis",
                "Phasor Plots",
            )
            os.makedirs(combined_dir, exist_ok=True)
            if plot_type == "Contour":
                path = os.path.join(
                    combined_dir, f"combined_contour_H{harmonic}.png"
                )
                _save_combined_contour(
                    items,
                    group_meta,
                    colors,
                    display,
                    legend,
                    path,
                    styles=self._contour_key_styles(items),
                    centers=centers,
                    center_color=merged_center_color,
                    dpi=self._export_dpi(),
                )
            else:
                single = len(items) == 1
                for key, real, imag in items:
                    if single:
                        path = os.path.join(
                            combined_dir,
                            f"combined_phasor_H{harmonic}.png",
                        )
                    else:
                        name = group_meta.get(key, (str(key), None))[0]
                        path = os.path.join(
                            combined_dir,
                            f"combined_phasor_{_safe_suffix(name)}"
                            f"_H{harmonic}.png",
                        )
                    _save_phasor_plot_png(
                        real,
                        imag,
                        display,
                        None,
                        path,
                        center=centers.get(key),
                        center_color=merged_center_color or colors.get(key),
                        dpi=self._export_dpi(),
                    )
                if not single:
                    all_path = os.path.join(
                        combined_dir,
                        f"combined_phasor_all_groups_H{harmonic}.png",
                    )
                    _save_grouped_overlay_plot(
                        items,
                        group_meta,
                        colors,
                        display,
                        legend,
                        all_path,
                        centers=centers,
                        dpi=self._export_dpi(),
                    )

        hist_source = (
            aggregate["hist_struct"] if streaming else aggregate["hist_data"]
        )
        for tab, labels in hist_source.items():
            stats_rows = []
            tab_name = self._TAB_FOLDERS.get(tab, "")
            tab_root = (
                os.path.join(
                    self._export_folder,
                    tab_name,
                    "Combined analysis",
                )
                if tab_name
                else os.path.join(self._export_folder, "Combined analysis")
            )
            hist_dir = os.path.join(tab_root, "Histograms")
            os.makedirs(hist_dir, exist_ok=True)
            for label, groups in labels.items():
                path = os.path.join(
                    hist_dir,
                    f"{tab}_{_safe_suffix(label)}_grouped_histogram.png",
                )
                rows = _save_grouped_histogram(
                    self._hist_group_items(
                        aggregate, tab, label, groups, streaming
                    ),
                    group_meta,
                    colors,
                    label,
                    self._group_config,
                    path,
                    dpi=self._export_dpi(),
                )
                stats_rows.extend(rows)
            if stats_rows:
                stats_dir = os.path.join(tab_root, "Statistics")
                os.makedirs(stats_dir, exist_ok=True)
                self._write_grouped_stats_csv(stats_dir, tab, stats_rows)

    @staticmethod
    def _contour_group_items(aggregate, harmonic, streaming):
        """Yield ``(key, real, imag)`` per group, one group at a time."""
        if streaming:
            for key in aggregate["contour_struct"][harmonic]:
                yield (
                    key,
                    aggregate["contour_store_r"].load((harmonic, key)),
                    aggregate["contour_store_i"].load((harmonic, key)),
                )
        else:
            for key, arrays in aggregate["contour_data"][harmonic].items():
                yield (
                    key,
                    np.concatenate([a[0] for a in arrays]),
                    np.concatenate([a[1] for a in arrays]),
                )

    @staticmethod
    def _group_centers(aggregate, harmonic):
        """Return ``{key: (g, s)}`` group centers for ``harmonic``.

        Each center is the intensity-weighted mean over all pooled pixels of the
        group (one center per group/merged key), reconstructed from the partial
        sums accumulated in :meth:`_accumulate_phasor_aggregate`.
        """
        centers = {}
        for key, acc in aggregate["centers_data"].get(harmonic, {}).items():
            weight, sum_real, sum_imag = acc
            if weight > 0:
                centers[key] = (sum_real / weight, sum_imag / weight)
        return centers

    @staticmethod
    def _hist_group_items(aggregate, tab, label, groups, streaming):
        """Yield ``(key, file_arrays)`` per group, one group at a time."""
        if streaming:
            for key, member_keys in groups.items():
                yield key, [
                    aggregate["hist_store"].load(member_key)
                    for member_key in member_keys
                ]
        else:
            for key, arrays in groups.items():
                yield key, list(arrays)

    def _write_grouped_stats_csv(self, target_dir, tab, stats_rows):
        """Write *tab*'s per-group statistics into *target_dir* as a CSV."""
        path = os.path.join(target_dir, f"{tab}_grouped_statistics.csv")
        with open(path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "Output",
                    "Group",
                    "Mean",
                    "Median",
                    "Std Dev",
                    "Center of Mass",
                    "N pixels",
                ]
            )
            for row in stats_rows:
                writer.writerow(
                    [
                        row["output"],
                        row["group"],
                        f"{row['mean']:.6f}",
                        f"{row['median']:.6f}",
                        f"{row['std']:.6f}",
                        f"{row['com']:.6f}",
                        row["n"],
                    ]
                )

    def _collect_plot_settings(self, mode="individual"):
        """Return the phasor-plot display dict for ``mode``.

        ``mode`` is ``"individual"`` (per-file plots) or ``"combined"`` (the
        pooled plot); the plot type, colormap, bins, log-scale and contour
        parameters are read from that mode's own controls, while the appearance
        (semicircle, background, center, frequency) and the zoom settings are
        shared between both modes.
        """
        controls = (
            self._plot_combined_controls
            if mode == "combined"
            else self._plot_individual_controls
        )
        return {
            "plot_type": controls["type"].currentData(),
            "semi_circle": not self.plot_semicircle_checkbox.isChecked(),
            "log_scale": controls["log"].isChecked(),
            "white_background": self.plot_white_bg_checkbox.isChecked(),
            "show_legend": self.plot_legend_checkbox.isChecked(),
            "colormap": controls["colormap"].currentText() or "jet",
            "bins": controls["bins"].value(),
            "contour_levels": controls["levels"].value(),
            "contour_linewidth": controls["linewidth"].value(),
            "marker_size": controls["marker_size"].value(),
            "marker_color": controls["marker_color"].color().name(),
            "marker_alpha": controls["marker_alpha"].value(),
            "show_center": self.plot_center_checkbox.isChecked(),
            "center_color": self.plot_center_color.color().name(),
            "frequency": float(self.plot_frequency_spin.text() or 0.0),
            "zoom": self._collect_zoom_settings(),
        }

    def _collect_zoom_settings(self):
        """Return the shared zoomed-section settings dict."""
        return {
            "export": self.plot_zoom_checkbox.isChecked(),
            "rectangle": (
                self.plot_zoom_checkbox.isChecked()
                and self.plot_zoom_rect_checkbox.isChecked()
            ),
            "xmin": self.plot_zoom_xmin.value(),
            "xmax": self.plot_zoom_xmax.value(),
            "ymin": self.plot_zoom_ymin.value(),
            "ymax": self.plot_zoom_ymax.value(),
        }

    def _collect_plot_jobs(self, pipeline):
        """Return phasor-plot render jobs for every enabled phasor plot.

        Individual (per-file) vs combined (merged/grouped) is decided per job in
        :meth:`_emit_file_outputs` from the global output-mode checkboxes, so
        each enabled plot is listed here regardless of those checkboxes.
        """
        jobs = []
        if (
            self.plot_individual_checkbox.isChecked()
            or self.plot_combined_checkbox.isChecked()
        ):
            jobs.append(
                {"suffix": "phasor_plot", "tab": None, "overlay": None}
            )
        if (
            pipeline.components is not None
            and self.components_plot_toggle.isChecked()
        ):
            jobs.append(
                {
                    "suffix": "components_phasor",
                    "tab": "components",
                    "overlay": {
                        "kind": "components",
                        "components": pipeline.components,
                    },
                }
            )
        if (
            pipeline.mapping is not None
            and self.mapping_plot_toggle.isChecked()
        ):
            jobs.extend(self._mapping_plot_jobs(pipeline.mapping))
        if pipeline.fret is not None and self.fret_plot_toggle.isChecked():
            jobs.append(
                {
                    "suffix": "fret_phasor",
                    "tab": "fret",
                    "overlay": {"kind": "fret", "fret": pipeline.fret},
                }
            )
        if (
            pipeline.selection is not None
            and self.selection_plot_toggle.isChecked()
        ):
            jobs.append(
                {
                    "suffix": "selection_phasor",
                    "tab": "selection",
                    "overlay": {
                        "kind": "selection",
                        "selection": pipeline.selection,
                    },
                }
            )
        return jobs

    def _mapping_plot_jobs(self, mapping):
        """Build mapping phasor-plot jobs (base + one per selected mesh)."""
        base_overlay = {
            "kind": "mapping",
            "color_by": mapping.get("color_by", "None"),
            "mesh": None,
            "mesh_colormap": mapping.get("mesh_colormap", "jet"),
            "mesh_alpha": mapping.get("mesh_alpha", 0.45),
            "mesh_phase_range": mapping.get("mesh_phase_range"),
            "mesh_modulation_range": mapping.get("mesh_modulation_range"),
            "mesh_clip_semicircle": mapping.get("mesh_clip_semicircle"),
        }
        jobs = [
            {
                "suffix": "mapping_phasor",
                "tab": "phasor_mapping",
                "overlay": base_overlay,
            }
        ]
        for mesh in mapping.get("meshes", []):
            overlay = dict(base_overlay)
            overlay["mesh"] = mesh
            jobs.append(
                {
                    "suffix": f"mapping_phasor_{mesh.lower()}_mesh",
                    "tab": "phasor_mapping",
                    "overlay": overlay,
                }
            )
        return jobs

    def _render_phasor_plot(self, layer, base_out, display, suffix, overlay):
        """Render and save a phasor-plot PNG per harmonic for ``layer``."""
        if display is None:
            return
        harmonics = np.atleast_1d(layer.metadata.get("harmonics"))
        mean = layer.metadata.get("original_mean")
        for harmonic in harmonics:
            real, imag = _select_harmonic_arrays(layer, int(harmonic))
            if real is None:
                continue
            center = None
            if display.get("show_center") and mean is not None:
                _, center_real, center_imag = phasor_center(mean, real, imag)
                center = (float(center_real), float(center_imag))
            plot_overlay = self._components_overlay_with_fraction_data(
                overlay, real, imag
            )
            _save_phasor_plot_png(
                real,
                imag,
                display,
                plot_overlay,
                f"{base_out}_{suffix}_H{int(harmonic)}.png",
                center=center,
                dpi=self._export_dpi(),
            )

    @staticmethod
    def _components_overlay_with_fraction_data(overlay, real, imag):
        """Attach the first-component fraction data to a components overlay."""
        if overlay is None or overlay.get("kind") != "components":
            return overlay
        components = overlay.get("components") or {}
        line_style = components.get("line_style") or {}
        if not line_style.get("show_fraction_histogram"):
            return overlay
        if components.get("analysis_type") != "linear":
            return overlay
        component_real = components.get("component_real")
        component_imag = components.get("component_imag")
        if component_real is None or component_imag is None:
            return overlay
        component_real = np.asarray(component_real)
        component_imag = np.asarray(component_imag)
        if component_real.ndim > 1:
            component_real = component_real[0]
            component_imag = component_imag[0]
        try:
            fraction = phasor_component_fraction(
                np.asarray(real),
                np.asarray(imag),
                component_real,
                component_imag,
            )
        except Exception:  # noqa: BLE001 - skip overlay if it can't compute
            return overlay
        new_overlay = dict(overlay)
        new_overlay["fraction_data"] = np.asarray(fraction, dtype=float)
        return new_overlay

    @staticmethod
    def _channel_label(layer, index):
        """Return a stable channel label for ``layer`` (settings, name or index)."""
        settings = (getattr(layer, "metadata", {}) or {}).get(
            "settings", {}
        ) or {}
        channel = settings.get("channel")
        if channel is not None:
            return channel
        name = getattr(layer, "name", "") or ""
        if "Channel" in name:
            label = name.split("Channel")[-1].strip().strip(":").strip()
            if label:
                return label
        return index

    def _emit_signal_outputs(self, path, results, ext, suffix, preserve):
        """Export / accumulate the signal profiles for one file's channels."""
        cfg = self._signal_export_cfg
        channels = []
        for index, (layer, _extra) in enumerate(results):
            profile = layer.metadata.get("_signal_profile")
            if profile is None:
                continue
            profile = self._normalize_signal(profile, cfg["normalize"])
            channels.append(
                (
                    self._channel_label(layer, index),
                    np.asarray(profile, dtype=float),
                )
            )
        if not channels:
            return
        if cfg["individual"]:
            self._save_signal_individual(path, ext, suffix, preserve, channels)
        if cfg["combined"]:
            group_key, group_name, group_color = self._group_for(
                os.path.basename(path)
            )
            entry = self._signal_combined.setdefault(
                group_key,
                {"name": group_name, "color": group_color, "channels": {}},
            )
            for label, profile in channels:
                entry["channels"].setdefault(label, []).append(profile)

    def _save_signal_individual(self, path, ext, suffix, preserve, channels):
        """Write the individual signal plot(s) for one file into 'Signal Plots'."""
        cfg = self._signal_export_cfg
        base = self._derive_output_path(
            path,
            ext,
            suffix,
            preserve,
            subfolder=os.path.join(
                "Individual image analysis", "Signal Plots"
            ),
        )
        title = os.path.basename(path)
        ylabel = self._signal_ylabel(cfg["normalize"])
        multichannel = len(channels) > 1
        if not multichannel or cfg["channel_mode"] == "together":
            if multichannel:
                lines = [
                    {
                        "label": f"Channel {label}",
                        "color": _channel_color(idx),
                        "y": profile,
                    }
                    for idx, (label, profile) in enumerate(channels)
                ]
                legend = True
            else:
                lines = [
                    {"label": None, "color": cfg["color"], "y": channels[0][1]}
                ]
                legend = False
            if cfg["png"]:
                _save_signal_lines_png(
                    lines,
                    f"{base}_signal.png",
                    title=title,
                    ylabel=ylabel,
                    legend=legend,
                    white_background=cfg["white_background"],
                    dpi=self._export_dpi(),
                )
            if cfg["csv"]:
                _save_signal_lines_csv(lines, f"{base}_signal.csv", ylabel)
        else:
            for label, profile in channels:
                lines = [{"label": None, "color": cfg["color"], "y": profile}]
                stem = f"{base}_Channel_{_safe_suffix(str(label))}_signal"
                if cfg["png"]:
                    _save_signal_lines_png(
                        lines,
                        f"{stem}.png",
                        title=f"{title} — Channel {label}",
                        ylabel=ylabel,
                        legend=False,
                        white_background=cfg["white_background"],
                        dpi=self._export_dpi(),
                    )
                if cfg["csv"]:
                    _save_signal_lines_csv(lines, f"{stem}.csv", ylabel)

    @staticmethod
    def _signal_band(profiles):
        """Return ``(mean, std, n)`` across ``profiles`` (aligned to min length)."""
        n_bins = min(p.size for p in profiles)
        if n_bins == 0:
            return None
        stacked = np.stack([p[:n_bins] for p in profiles], axis=0)
        return (
            np.nanmean(stacked, axis=0),
            np.nanstd(stacked, axis=0),
            len(profiles),
        )

    def _write_signal_combined(self):
        """Write the combined mean ± SD signal plot(s) (per group / channel)."""
        if not self._signal_combined:
            return
        cfg = self._signal_export_cfg
        ylabel = self._signal_ylabel(cfg["normalize"])

        channel_labels = []
        for entry in self._signal_combined.values():
            for label in entry["channels"]:
                if label not in channel_labels:
                    channel_labels.append(label)
        if not channel_labels:
            return
        multichannel = len(channel_labels) > 1
        single_group = len(self._signal_combined) <= 1
        combined_dir = os.path.join(
            self._export_folder, "Combined analysis", "Signal Plots"
        )
        os.makedirs(combined_dir, exist_ok=True)

        if multichannel and cfg["channel_mode"] == "separate":
            for label in channel_labels:
                bands = []
                for entry in self._signal_combined.values():
                    profiles = entry["channels"].get(label)
                    if not profiles:
                        continue
                    band = self._signal_band(profiles)
                    if band is None:
                        continue
                    mean, std, n = band
                    bands.append(
                        {
                            "label": f"{entry['name']} (n={n})",
                            "color": entry["color"] or None,
                            "linestyle": "-",
                            "mean": mean,
                            "std": std,
                        }
                    )
                if not bands:
                    continue
                stem = os.path.join(
                    combined_dir,
                    f"combined_signal_Channel_{_safe_suffix(str(label))}",
                )
                if cfg["png"]:
                    _save_signal_bands_png(
                        bands,
                        f"{stem}.png",
                        title=f"Channel {label}",
                        white_background=cfg["white_background"],
                        legend=cfg["legend"],
                        ylabel=ylabel,
                        dpi=self._export_dpi(),
                    )
                if cfg["csv"]:
                    _save_signal_bands_csv(bands, f"{stem}.csv", ylabel)
            return

        bands = []
        for entry in self._signal_combined.values():
            for idx, label in enumerate(channel_labels):
                profiles = entry["channels"].get(label)
                if not profiles:
                    continue
                band = self._signal_band(profiles)
                if band is None:
                    continue
                mean, std, n = band
                if not multichannel:
                    line_label = f"{entry['name']} (n={n})"
                    color = entry["color"] or None
                    linestyle = "-"
                elif single_group:
                    line_label = f"Channel {label} (n={n})"
                    color = _channel_color(idx)
                    linestyle = "-"
                else:
                    line_label = f"{entry['name']} · Ch{label} (n={n})"
                    color = entry["color"] or None
                    linestyle = _CHANNEL_LINESTYLES[
                        idx % len(_CHANNEL_LINESTYLES)
                    ]
                bands.append(
                    {
                        "label": line_label,
                        "color": color,
                        "linestyle": linestyle,
                        "mean": mean,
                        "std": std,
                    }
                )
        if not bands:
            return
        stem = os.path.join(combined_dir, "combined_signal")
        if cfg["png"]:
            _save_signal_bands_png(
                bands,
                f"{stem}.png",
                white_background=cfg["white_background"],
                legend=cfg["legend"],
                ylabel=ylabel,
                dpi=self._export_dpi(),
            )
        if cfg["csv"]:
            _save_signal_bands_csv(bands, f"{stem}.csv", ylabel)


def _masked_signal_mean(full, axis, mask):
    """Return the per-pixel mean signal over the mask along ``axis``."""
    arr = np.asarray(full, dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1)
    if axis is None:
        axis = arr.ndim - 1
    axis = int(axis) % arr.ndim
    moved = np.moveaxis(arr, axis, 0)
    flat = moved.reshape(moved.shape[0], -1)
    valid = None
    if mask is not None:
        m = np.asarray(mask["array"])
        invalid = m > 0 if mask.get("invert") else m <= 0
        keep = ~invalid
        if keep.shape == moved.shape[1:]:
            valid = keep.reshape(-1)
    if valid is not None and valid.any():
        return np.nanmean(flat[:, valid], axis=1)
    return np.nanmean(flat, axis=1)


# Line styles cycled to distinguish channels of the same group when combined
# channel bands are overlaid ("Together" mode with more than one group).
_CHANNEL_LINESTYLES = ["-", "--", ":", "-."]


def _channel_color(index):
    """Return a distinct color for channel ``index`` (tab10 cycle)."""
    import matplotlib.pyplot as plt

    return plt.cm.tab10(index % 10)


def _save_signal_lines_png(
    lines,
    path,
    title=None,
    ylabel="Mean signal per pixel",
    legend=False,
    white_background=True,
    dpi=300,
):
    """Render one or more 1-D signal lines to ``path`` as a PNG.

    ``lines`` is a list of dicts with ``y`` and optional ``label``/``color``/
    ``linestyle``.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    if white_background:
        fig.set_facecolor("white")
        ax.set_facecolor("white")
    for line in lines:
        y = np.asarray(line["y"], dtype=float).ravel()
        ax.plot(
            np.arange(y.size),
            y,
            color=line.get("color"),
            linewidth=1.5,
            linestyle=line.get("linestyle", "-"),
            label=line.get("label"),
        )
    ax.set_xlabel("Histogram / spectral bin")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=9)
    ax.margins(x=0)
    if legend:
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _csv_column_label(label, fallback):
    """Return a safe, human-readable CSV column name for a plot ``label``."""
    text = str(label) if label not in (None, "") else fallback
    return text.replace("\n", " ").strip()


def _save_signal_lines_csv(lines, path, value_header="value"):
    """Write signal ``lines`` (one ``y`` column each) to ``path`` as CSV.

    The first column is the histogram / spectral bin index; each subsequent
    column is one line's values, aligned to the shortest line.
    """
    ys = [np.asarray(line["y"], dtype=float).ravel() for line in lines]
    if not ys:
        return
    n_bins = min(y.size for y in ys)
    if n_bins == 0:
        return
    headers = ["bin"]
    for idx, line in enumerate(lines):
        base = _csv_column_label(line.get("label"), value_header)
        headers.append(base if len(lines) == 1 else f"{base} ({idx + 1})")
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for i in range(n_bins):
            writer.writerow([i] + [f"{y[i]:.6g}" for y in ys])


def _save_signal_bands_csv(bands, path, value_header="value"):
    """Write mean ± SD ``bands`` to ``path`` as CSV.

    The first column is the histogram / spectral bin index; each band adds a
    ``<label> mean`` and ``<label> std`` column, aligned to the shortest band.
    """
    means = [np.asarray(b["mean"], dtype=float).ravel() for b in bands]
    stds = [np.asarray(b["std"], dtype=float).ravel() for b in bands]
    if not means:
        return
    n_bins = min(m.size for m in means)
    if n_bins == 0:
        return
    headers = ["bin"]
    for idx, band in enumerate(bands):
        base = _csv_column_label(
            band.get("label"), f"{value_header} {idx + 1}"
        )
        headers.extend([f"{base} mean", f"{base} std"])
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for i in range(n_bins):
            row = [i]
            for mean, std in zip(means, stds, strict=True):
                row.extend([f"{mean[i]:.6g}", f"{std[i]:.6g}"])
            writer.writerow(row)


def _save_signal_bands_png(
    bands,
    path,
    title=None,
    white_background=True,
    legend=True,
    ylabel="Mean signal per pixel",
    dpi=300,
):
    """Render mean ± SD signal bands overlaid to ``path``.

    ``bands`` is a list of dicts with ``mean``/``std`` and optional ``label``/
    ``color``/``linestyle``.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    if white_background:
        fig.set_facecolor("white")
        ax.set_facecolor("white")
    for band in bands:
        mean = np.asarray(band["mean"], dtype=float)
        std = np.asarray(band["std"], dtype=float)
        x = np.arange(mean.size)
        color = band.get("color")
        ax.plot(
            x,
            mean,
            color=color,
            linewidth=1.5,
            linestyle=band.get("linestyle", "-"),
            label=band.get("label"),
        )
        ax.fill_between(
            x, mean - std, mean + std, color=color, alpha=0.25, linewidth=0
        )
    ax.set_xlabel("Histogram / spectral bin")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=9)
    ax.margins(x=0)
    if legend:
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _save_plot_with_zoom(plot, path, zoom, dpi=300):
    """Save ``plot`` to ``path`` and optionally a zoomed-in crop.

    ``zoom`` is ``None`` or a dict with ``export`` (also write a ``*_zoom.png``
    cropped to ``xmin/xmax/ymin/ymax``) and ``rectangle`` (outline the zoom
    region on the full plot). The rectangle is drawn only on the full plot, not
    on the cropped one. The caller still owns closing the figure. ``dpi``
    controls the resolution of the saved PNG(s).
    """
    if not zoom or not (zoom.get("export") or zoom.get("rectangle")):
        plot.save(path, dpi=dpi)
        return

    xmin, xmax = zoom["xmin"], zoom["xmax"]
    ymin, ymax = zoom["ymin"], zoom["ymax"]

    rect = None
    if zoom.get("rectangle"):
        from matplotlib.patches import Rectangle

        rect = Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            fill=False,
            edgecolor="black",
            linewidth=1,
            zorder=30,
        )
        plot.ax.add_patch(rect)

    full_xlim = plot.ax.get_xlim()
    full_ylim = plot.ax.get_ylim()
    plot.save(path, dpi=dpi)

    if zoom.get("export"):
        if rect is not None:
            rect.remove()
        plot.ax.set_xlim(xmin, xmax)
        plot.ax.set_ylim(ymin, ymax)
        if path.lower().endswith(".png"):
            zoom_path = f"{path[:-4]}_zoom.png"
        else:
            zoom_path = f"{path}_zoom"
        plot.save(zoom_path, dpi=dpi)
        plot.ax.set_xlim(full_xlim)
        plot.ax.set_ylim(full_ylim)


def _save_phasor_plot_png(
    real,
    imag,
    display,
    overlay,
    path,
    center=None,
    center_color=None,
    dpi=300,
):
    """Render a phasor plot of ``(real, imag)`` to ``path`` via PhasorPlot.

    ``display`` holds the plot-settings dict; ``overlay`` is ``None`` or a dict
    describing tab-specific features to draw (components, cursors, FRET donor
    trajectory); ``center`` is an optional ``(g, s)`` phasor-center marker, and
    ``center_color`` overrides ``display['center_color']`` for it (used to color
    a grouped center by its group color).
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from phasorpy.plot import PhasorPlot

    real = np.asarray(real).ravel()
    imag = np.asarray(imag).ravel()
    valid = np.isfinite(real) & np.isfinite(imag)
    real = real[valid]
    imag = imag[valid]

    frequency = display.get("frequency") or None
    plot = PhasorPlot(
        allquadrants=not display.get("semi_circle", True),
        frequency=frequency,
    )
    plot.ax.set_title("")
    if display.get("white_background"):
        plot.ax.set_facecolor("white")
        plot.fig.set_facecolor("white")

    mapping_overlay = (
        overlay if overlay and overlay.get("kind") == "mapping" else None
    )

    # Optional phase/modulation mesh field behind the data.
    if mapping_overlay and mapping_overlay.get("mesh"):
        _draw_phase_modulation_mesh(
            plot,
            mapping_overlay["mesh"],
            mapping_overlay.get("mesh_colormap") or "jet",
            mapping_overlay.get("mesh_alpha", 0.45),
            display.get("semi_circle", True),
            phase_range=mapping_overlay.get("mesh_phase_range"),
            modulation_range=mapping_overlay.get("mesh_modulation_range"),
            clip_semicircle=mapping_overlay.get("mesh_clip_semicircle"),
            dpi=dpi,
        )

    plot_type = display.get("plot_type", "Histogram")

    # Optional coloring of the data by a mapping metric (Phase / Modulation).
    # Mirrors ``PhasorMappingWidget._apply_histogram_coloring``: instead of
    # always drawing a scatter, the *selected* plot type is colored by the
    # metric (a scatter point-by-point, a density / contour by per-bin median).
    metric = None
    metric_cmap = None
    metric_range = None
    if (
        mapping_overlay
        and mapping_overlay.get("color_by", "None") != "None"
        and real.size
    ):
        with np.errstate(divide="ignore", invalid="ignore"):
            phase, modulation = phasor_to_polar(real, imag)
        color_by = mapping_overlay["color_by"]
        metric = phase if color_by == "Phase" else modulation
        metric_cmap = mapping_overlay.get("mesh_colormap") or "jet"
        # Use the same color range as the phase/modulation mesh so the colored
        # data and the mesh represent identical values with identical colors.
        metric_range = (
            mapping_overlay.get("mesh_phase_range")
            if color_by == "Phase"
            else mapping_overlay.get("mesh_modulation_range")
        )

    if real.size and plot_type != "None":
        if metric is not None:
            _color_plot_by_metric(
                plot,
                real,
                imag,
                metric,
                metric_cmap,
                display,
                plot_type,
                value_range=metric_range,
            )
        elif plot_type == "Scatter":
            plot.plot(
                real,
                imag,
                fmt="o",
                markersize=display.get("marker_size", 5),
                color=display.get("marker_color") or None,
                alpha=display.get("marker_alpha", 0.3),
                zorder=5,
            )
        elif plot_type == "Contour":
            cmap = (
                resolve_colormap_by_name(display.get("colormap"))
                if display.get("colormap")
                else None
            )
            _draw_widget_contour(plot, real, imag, display, cmap=cmap)
        else:
            kwargs = {"bins": display.get("bins", 300), "zorder": 5}
            if display.get("colormap"):
                kwargs["cmap"] = display["colormap"]
            if display.get("log_scale"):
                kwargs["norm"] = LogNorm()
            plot.hist2d(real, imag, **kwargs)

    if overlay and not mapping_overlay:
        _add_phasor_overlay(plot, overlay)

    if center is not None:
        plot.plot(
            [center[0]],
            [center[1]],
            fmt="o",
            color=center_color or display.get("center_color", "red"),
            markersize=9,
        )

    _save_plot_with_zoom(plot, path, display.get("zoom"), dpi=dpi)
    plt.close(plot.fig)


def _colormap_color_list(name, samples=256):
    """Return a list of RGBA colors sampled from a named colormap.

    Mirrors the per-fraction ``fractions_colormap`` color list the interactive
    Components tab feeds to :func:`draw_components_overlay`, so the exported
    colormap line and colormap-end dot colors match. Returns ``None`` if the
    colormap cannot be resolved.
    """
    cmap = resolve_colormap_by_name(name)
    if cmap is None:
        return None
    return [cmap(t) for t in np.linspace(0.0, 1.0, samples)]


def _component_dot_colors(
    analysis_type,
    colormap_names,
    fractions_colormap,
    contrast_limits,
    show_colormap_line,
    default_color,
):
    """Return per-component dot colors matching the interactive Components tab.

    Mirrors ``ComponentsWidget._get_component_colors_for_count``: when the
    overlay colormap is off, all dots use the default line color. For a
    2-component Linear Projection the dots take the colormap-end colors (scaled
    by the contrast limits); otherwise each dot uses the top color of its own
    component colormap.
    """
    count = len(colormap_names)
    if not show_colormap_line:
        return [default_color] * count

    if (
        analysis_type == "Linear Projection"
        and count == 2
        and fractions_colormap
    ):
        vmin, vmax = contrast_limits or (0.0, 1.0)
        n = len(fractions_colormap)
        if vmax > vmin:
            idx1 = int(((1.0 - vmin) / (vmax - vmin)) * (n - 1))
            idx2 = int(((0.0 - vmin) / (vmax - vmin)) * (n - 1))
            idx1 = max(0, min(n - 1, idx1))
            idx2 = max(0, min(n - 1, idx2))
            return [fractions_colormap[idx1], fractions_colormap[idx2]]
        return [fractions_colormap[-1], fractions_colormap[0]]

    colors = []
    for name in colormap_names:
        cmap = resolve_colormap_by_name(name)
        if cmap is not None:
            colors.append(cmap(1.0))
        else:
            colors.append(default_color)
    return colors


def _draw_phase_modulation_mesh(
    plot,
    kind,
    colormap,
    alpha,
    semicircle,
    phase_range=None,
    modulation_range=None,
    clip_semicircle=None,
    dpi=300,
):
    """Draw a phase or modulation colored field behind the phasor data.

    Delegates to :func:`napari_phasors.phasor_mapping_tab.draw_phasor_mesh` so
    the exported mesh matches the interactive Phasor Mapping tab exactly
    (smoothed alpha edges, correct color scaling and a 1:1 data aspect). The
    optional phase/modulation ranges restrict which cells are shown, mirroring
    the interactive range sliders. ``clip_semicircle`` defaults to the plot
    geometry (clip in semicircle mode).
    """
    from .phasor_mapping_tab import draw_phasor_mesh

    if clip_semicircle is None:
        clip_semicircle = semicircle
    plot.fig.set_dpi(dpi)
    draw_phasor_mesh(
        plot.ax,
        kind,
        semicircle=semicircle,
        colormap=colormap,
        alpha=alpha,
        phase_range=phase_range,
        modulation_range=modulation_range,
        clip_semicircle=clip_semicircle,
        resolution=1000,
    )


def _add_phasor_overlay(plot, overlay):
    """Draw a tab-specific overlay onto an existing :class:`PhasorPlot`."""
    kind = overlay["kind"]

    if kind == "components":
        from .components_tab import draw_components_overlay

        components = overlay["components"]
        # The pipeline stores "linear"/"fit"; draw_components_overlay expects
        # the interactive labels to decide colormap-line vs polygon rendering.
        analysis_type = (
            "Linear Projection"
            if components.get("analysis_type") == "linear"
            else "Component Fit"
        )
        line_style = components.get("line_style") or {}
        label_style = components.get("label_style") or {}
        fractions_colormap = components.get("fractions_colormap")
        contrast = components.get("colormap_contrast_limits") or (0.0, 1.0)

        colors = _component_dot_colors(
            analysis_type,
            components.get("colormaps") or [],
            fractions_colormap,
            contrast,
            line_style.get("show_colormap_line", True),
            line_style.get("default_component_color", "dimgray"),
        )
        settings = {
            "analysis_type": analysis_type,
            "line_offset": line_style.get("line_offset", 0.0),
            "line_width": line_style.get("line_width", 3.0),
            "line_alpha": line_style.get("line_alpha", 1.0),
            "show_colormap_line": line_style.get("show_colormap_line", True),
            "show_dots": line_style.get("show_component_dots", True),
            "default_component_color": line_style.get(
                "default_component_color", "dimgray"
            ),
            "fractions_colormap": fractions_colormap,
            "colormap_contrast_limits": contrast,
            "colormap_gamma": line_style.get("colormap_gamma", 1.0),
            "label_fontsize": label_style.get("fontsize", 10),
            "label_fontweight": (
                "bold" if label_style.get("bold") else "normal"
            ),
            "label_fontstyle": (
                "italic" if label_style.get("italic") else "normal"
            ),
            "label_color": label_style.get("color"),
            "show_labels": label_style.get("show_labels", False),
            # Fraction histogram overlay (Linear Projection); the fraction data
            # is injected per-harmonic by ``_render_phasor_plot``.
            "show_fraction_histogram": line_style.get(
                "show_fraction_histogram", False
            ),
            "histogram_overlay_height": line_style.get(
                "histogram_overlay_height", 0.3
            ),
            "histogram_offset": line_style.get("histogram_offset", 0.0),
            "histogram_alpha": line_style.get("histogram_alpha", 0.75),
            "fraction_data": overlay.get("fraction_data"),
        }
        # Multi-harmonic fits store 2-D component arrays; the overlay draws a
        # single harmonic, so use the primary harmonic's locations.
        component_real = components["component_real"]
        component_imag = components["component_imag"]
        if np.ndim(component_real) > 1:
            component_real = np.asarray(component_real)[0]
            component_imag = np.asarray(component_imag)[0]
        draw_components_overlay(
            plot.ax,
            component_real,
            component_imag,
            components.get("names"),
            colors,
            analysis_type,
            settings,
        )

    elif kind == "selection":
        from .selection_tab import draw_selection_overlay

        selection = overlay["selection"]
        # Cluster regions are data-dependent, so no static cursors to overlay
        if selection.get("mode") == "cluster":
            return
        draw_selection_overlay(plot.ax, selection["cursors"], mode="cursor")

    elif kind == "fret":
        from .fret_tab import draw_fret_trajectory_overlay

        fret = overlay["fret"]
        efficiencies = np.linspace(0, 1, 500)
        neighbor_real, neighbor_imag = phasor_from_fret_donor(
            fret["frequency"] * fret["harmonic"],
            fret["donor_lifetime"],
            fret_efficiency=efficiencies,
            donor_background=fret["donor_background"],
            background_real=fret["background_real"],
            background_imag=fret["background_imag"],
            donor_fretting=fret["donor_fretting"],
        )
        settings = {
            "use_colormap": True,
            "fret_colormap": fret.get("colormap"),
            "colormap_contrast_limits": fret.get("contrast_limits", (0, 1)),
        }
        draw_fret_trajectory_overlay(
            plot.ax, neighbor_real, neighbor_imag, efficiencies, settings
        )


def _color_plot_by_metric(
    plot, real, imag, metric, cmap, display, plot_type, value_range=None
):
    """Color the phasor plot by a per-point metric (Phase / Modulation).

    Mirrors ``PhasorMappingWidget._apply_histogram_coloring`` for the static
    export: a scatter plot is colored point-by-point, while a density or
    contour plot is colored by the per-bin median metric (with the contour
    lines drawn on top for the contour type). Empty bins stay transparent.
    ``value_range`` (``(min, max)``) pins the color scale so it matches the
    phase/modulation mesh and the exported metric image.
    """
    import warnings

    from scipy.stats import binned_statistic_2d

    ax = plot.ax
    real = np.asarray(real).ravel()
    imag = np.asarray(imag).ravel()
    metric = np.asarray(metric).ravel()
    finite = np.isfinite(real) & np.isfinite(imag) & np.isfinite(metric)
    real, imag, metric = real[finite], imag[finite], metric[finite]
    if not real.size:
        return

    resolved = cmap
    if resolved is None or isinstance(resolved, str):
        resolved = resolve_colormap_by_name(resolved or "jet")
    if resolved is None:
        resolved = resolve_colormap_by_name("jet")

    vmin = vmax = None
    if value_range is not None and value_range[0] < value_range[1]:
        vmin, vmax = value_range

    if plot_type == "Scatter":
        ax.scatter(
            real,
            imag,
            c=metric,
            cmap=resolved,
            vmin=vmin,
            vmax=vmax,
            s=display.get("marker_size", 5),
            alpha=display.get("marker_alpha", 0.6),
            zorder=5,
        )
        return

    # Density / contour: color by the per-bin median metric.
    range_xlim = ax.get_xlim()
    range_ylim = ax.get_ylim()
    bins = display.get("bins", 300)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        counts, xedges, yedges = np.histogram2d(
            real, imag, bins=bins, range=[range_xlim, range_ylim]
        )
        stat, _, _, _ = binned_statistic_2d(
            real, imag, metric, statistic="median", bins=[xedges, yedges]
        )

    if plot_type == "Contour":
        _draw_widget_contour(plot, real, imag, display, cmap=resolved)

    mask = ~np.isfinite(stat) | (counts <= 0)
    stat_masked = np.ma.array(stat.T, mask=mask.T)
    mesh_cmap = resolved.copy()
    mesh_cmap.set_bad((0, 0, 0, 0))
    ax.pcolormesh(
        xedges,
        yedges,
        stat_masked,
        cmap=mesh_cmap,
        vmin=vmin,
        vmax=vmax,
        shading="auto",
        zorder=4,
    )


def _draw_widget_contour(
    plot, real, imag, display, cmap=None, color=None, name="contour"
):
    """Draw a log-normed 2-D-histogram contour onto ``plot``.

    Mirrors ``PlotterWidget._update_contour_plot`` /
    ``_compute_contour_histogram`` so the exported contour matches the
    interactive Phasor Plot widget instead of phasorpy's KDE-based
    ``PhasorPlot.contour``. Pass either ``cmap`` (colormap mode) or ``color``
    (solid mode). Returns the representative color for a legend handle, or
    ``None`` when there was nothing to draw.
    """
    real = np.asarray(real).ravel()
    imag = np.asarray(imag).ravel()
    finite = np.isfinite(real) & np.isfinite(imag)
    real = real[finite]
    imag = imag[finite]
    if not real.size:
        return None

    ax = plot.ax
    range_xlim = ax.get_xlim()
    range_ylim = ax.get_ylim()
    bins = display.get("bins", 300)
    aspect = (range_xlim[1] - range_xlim[0]) / (range_ylim[1] - range_ylim[0])
    if aspect > 1:
        bins = (bins, max(int(bins / aspect), 1))
    else:
        bins = (max(int(bins * aspect), 1), bins)

    h, xedges, yedges = np.histogram2d(
        real, imag, bins=bins, range=[range_xlim, range_ylim]
    )
    xcenters = xedges[:-1] + ((xedges[1] - xedges[0]) / 2.0)
    ycenters = yedges[:-1] + ((yedges[1] - yedges[0]) / 2.0)
    h = h.astype(float)
    h[h <= 0] = np.nan

    if cmap is None:
        cmap = make_solid_contour_cmap(name, color or "tab:blue")
        legend_color = normalize_rgb(color or "tab:blue")
    else:
        legend_color = cmap(0.6)

    ax.contour(
        xcenters,
        ycenters,
        h.T,
        levels=display.get("contour_levels", 7),
        linewidths=display.get("contour_linewidth", 1.5),
        cmap=cmap,
        norm="log",
        zorder=5,
    )
    return legend_color


def _save_grouped_overlay_plot(
    group_items,
    group_meta,
    colors,
    display,
    legend,
    path,
    overlay=None,
    centers=None,
    dpi=300,
):
    """Save one phasor plot overlaying every group (group-colored scatter).

    Complements the per-group plots: each group's pooled coordinates are drawn
    as a scatter in its group color, with a legend, so all groups are visible
    in a single phasor plot. Any static analysis ``overlay`` (phase/modulation
    mesh, component line, FRET trajectory) is drawn once behind the data.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from phasorpy.plot import PhasorPlot

    plot = PhasorPlot(
        allquadrants=not display.get("semi_circle", True),
        frequency=display.get("frequency") or None,
    )
    plot.ax.set_title("")
    if display.get("white_background"):
        plot.ax.set_facecolor("white")
        plot.fig.set_facecolor("white")

    mapping_overlay = (
        overlay if overlay and overlay.get("kind") == "mapping" else None
    )
    if mapping_overlay and mapping_overlay.get("mesh"):
        _draw_phase_modulation_mesh(
            plot,
            mapping_overlay["mesh"],
            mapping_overlay.get("mesh_colormap") or "jet",
            mapping_overlay.get("mesh_alpha", 0.45),
            display.get("semi_circle", True),
            phase_range=mapping_overlay.get("mesh_phase_range"),
            modulation_range=mapping_overlay.get("mesh_modulation_range"),
            clip_semicircle=mapping_overlay.get("mesh_clip_semicircle"),
            dpi=dpi,
        )

    marker_size = display.get("marker_size", 5)
    alpha = display.get("marker_alpha", 0.3)
    centers = centers or {}
    handles = []
    for key, real, imag in group_items:
        real = np.asarray(real).ravel()
        imag = np.asarray(imag).ravel()
        finite = np.isfinite(real) & np.isfinite(imag)
        real, imag = real[finite], imag[finite]
        if not real.size:
            continue
        name = group_meta.get(key, (str(key), None))[0]
        color = colors.get(key)
        plot.ax.scatter(
            real, imag, s=marker_size, color=color, alpha=alpha, zorder=5
        )
        handles.append(
            Line2D([0], [0], marker="o", linestyle="", color=color, label=name)
        )
        if key in centers:
            gc, sc = centers[key]
            plot.plot(
                [gc], [sc], fmt="o", color=color, markersize=9, zorder=10
            )

    if overlay and not mapping_overlay:
        _add_phasor_overlay(plot, overlay)

    if legend and handles:
        plot.ax.legend(handles=handles, loc="upper right")
    _save_plot_with_zoom(plot, path, display.get("zoom"), dpi=dpi)
    plt.close(plot.fig)


def _save_combined_contour(
    group_items,
    group_meta,
    colors,
    display,
    legend,
    path,
    styles=None,
    centers=None,
    center_color=None,
    overlay=None,
    dpi=300,
):
    """Save a combined contour phasor plot with one contour per group.

    ``group_items`` yields ``(key, real, imag)`` one group at a time, so the
    caller controls whether the data comes from memory or disk. ``styles`` is
    an optional ``{key: {"mode", "colormap", "color"}}`` mapping (from the
    Contour Layer Settings dialog): when a key's style is ``colormap`` the
    contour is colored with that colormap, otherwise a solid color is used,
    mirroring ``PlotterWidget._update_contour_plot``. ``overlay`` (if given)
    draws the analysis overlay (phase/modulation mesh, component line, FRET
    trajectory) behind the contours.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from phasorpy.plot import PhasorPlot

    styles = styles or {}

    plot = PhasorPlot(
        allquadrants=not display.get("semi_circle", True),
        frequency=display.get("frequency") or None,
    )
    plot.ax.set_title("")
    if display.get("white_background"):
        plot.ax.set_facecolor("white")
        plot.fig.set_facecolor("white")

    mapping_overlay = (
        overlay if overlay and overlay.get("kind") == "mapping" else None
    )
    if mapping_overlay and mapping_overlay.get("mesh"):
        _draw_phase_modulation_mesh(
            plot,
            mapping_overlay["mesh"],
            mapping_overlay.get("mesh_colormap") or "jet",
            mapping_overlay.get("mesh_alpha", 0.45),
            display.get("semi_circle", True),
            phase_range=mapping_overlay.get("mesh_phase_range"),
            modulation_range=mapping_overlay.get("mesh_modulation_range"),
            clip_semicircle=mapping_overlay.get("mesh_clip_semicircle"),
            dpi=dpi,
        )

    handles = []
    centers = centers or {}
    for key, real, imag in group_items:
        real = np.asarray(real)
        imag = np.asarray(imag)
        if not real.size:
            continue
        name = group_meta.get(key, (str(key), None))[0]
        style = styles.get(key) or {}
        cmap = None
        if style.get("mode") == "colormap":
            cmap = resolve_colormap_by_name(style.get("colormap"))
        if cmap is not None:
            legend_color = _draw_widget_contour(
                plot, real, imag, display, cmap=cmap, name=f"contour_{name}"
            )
        else:
            color = style.get("color") or colors.get(key)
            legend_color = _draw_widget_contour(
                plot, real, imag, display, color=color, name=f"contour_{name}"
            )
        if legend_color is None:
            continue
        handles.append(Line2D([0], [0], color=legend_color, label=name))
        # One phasor center for the group as a whole, drawn on this same plot.
        if key in centers:
            gc, sc = centers[key]
            plot.plot(
                [gc],
                [sc],
                fmt="o",
                color=center_color or colors.get(key),
                markersize=9,
                zorder=10,
            )
    if overlay and not mapping_overlay:
        _add_phasor_overlay(plot, overlay)
    if legend and handles:
        plot.ax.legend(handles=handles, loc="upper right")
    _save_plot_with_zoom(plot, path, display.get("zoom"), dpi=dpi)
    plt.close(plot.fig)


def _save_grouped_histogram(
    group_items, group_meta, colors, label, config, path, dpi=300
):
    """Save a combined histogram and return exact per-group statistics rows.

    ``group_items`` yields ``(key, file_arrays)`` one group at a time, where
    ``file_arrays`` is a list of the per-file 1-D arrays belonging to that
    group. Statistics are computed on the full (exact) pooled values via
    :func:`_compute_stats`, so the result is identical whether the data came
    from memory or disk.

    Rendering honors the global display mode:

    - ``"Grouped"`` draws one mean curve per group; when ``show_sd`` is on and
      the group has more than one file, a shaded standard-deviation band (the
      spread of the per-file histograms) is drawn around the mean, mirroring
      the interactive Plotter's grouped histogram.
    - ``"Merged"`` / ``"Individual layers"`` draw one pooled curve per group
      key, as before.
    """
    mode = config.get("mode", "Merged")
    stats_rows = []

    def _pooled(file_arrays):
        parts = [
            np.asarray(a)[np.isfinite(a)] for a in file_arrays if np.size(a)
        ]
        return parts, (
            np.concatenate(parts) if parts else np.empty(0, dtype=float)
        )

    if mode == "Grouped":
        datasets = {}
        assignments = {}
        group_names = {}
        group_colors = {}
        for gid, (key, file_arrays) in enumerate(group_items, start=1):
            name = group_meta.get(key, (str(key), None))[0]
            parts, pooled = _pooled(file_arrays)
            row = _compute_stats(pooled, 100)
            row["output"] = label
            row["group"] = name
            stats_rows.append(row)
            group_names[gid] = name
            color = colors.get(key)
            if color is not None:
                group_colors[gid] = color
            # Each file becomes its own dataset assigned to this group so the
            # renderer can compute the across-file mean curve and SD band.
            for file_index, valid in enumerate(parts):
                if valid.size:
                    member_label = f"{gid}:{name}:{file_index}"
                    datasets[member_label] = valid
                    assignments[member_label] = gid

        if datasets:
            hw = _new_export_histogram(config, label)
            hw.display_mode = "Grouped"
            hw._group_assignments = assignments
            hw._group_names = group_names
            hw._group_colors = group_colors
            hw.update_multi_data(datasets)
            # ``update_multi_data`` auto-enables SD when several datasets are
            # added; override it so the configured choice wins.
            hw.show_sd = config.get("show_sd", True)
            _save_export_histogram(hw, path, dpi=dpi)
        return stats_rows

    # Merged / Individual layers: one pooled curve per group key.
    datasets = {}
    for key, file_arrays in group_items:
        name = group_meta.get(key, (str(key), None))[0]
        _parts, pooled = _pooled(file_arrays)
        row = _compute_stats(pooled, 100)
        row["output"] = label
        row["group"] = name
        stats_rows.append(row)
        if pooled.size:
            datasets[name] = pooled

    if datasets:
        hw = _new_export_histogram(config, label)
        hw.display_mode = "Individual layers"
        layer_colors = {}
        for key, color in colors.items():
            gname = group_meta.get(key, (str(key), None))[0]
            layer_colors[gname] = color
        hw._layer_colors = layer_colors
        hw.update_multi_data(datasets)
        _save_export_histogram(hw, path, dpi=dpi)

    return stats_rows


def _new_export_histogram(config, label):
    """Return a :class:`HistogramWidget` configured from ``config``."""
    hw = HistogramWidget()
    hw.white_background = config.get("white_background", False)
    hw._smooth_curves = config.get("smooth_curves", True)
    hw._central_tendency = config.get("central_tendency", "None")
    hw._show_legend = config.get("show_legend", True)
    hw.xlabel = label
    return hw


def _save_export_histogram(hw, path, dpi=300):
    """Style a histogram for export and save it to ``path``."""
    hw._style_axes(export_mode=True)
    hw.fig.canvas.draw_idle()
    use_transparent = not hw._white_background
    hw.fig.savefig(
        path,
        dpi=dpi,
        bbox_inches='tight',
        transparent=use_transparent,
        facecolor='white' if hw._white_background else 'none',
    )


class _SpillStore:
    """Append-only on-disk store of float32 arrays, keyed by hashable keys.

    Used for bounded-memory aggregation: each group's data is spilled to a
    temporary file during processing and read back one group at a time when
    rendering/reducing, so statistics are computed on the full exact data
    without holding everything in RAM.
    """

    def __init__(self, root):
        """Create the store, backing it with files under the *root* directory."""
        self._root = root
        os.makedirs(root, exist_ok=True)
        self._paths = {}

    def append(self, key, array):
        """Append *array* to *key*'s data, creating its file on first use."""
        path = self._paths.get(key)
        if path is None:
            path = os.path.join(self._root, f"{len(self._paths)}.f64")
            self._paths[key] = path
        with open(path, "ab") as handle:
            np.asarray(array, dtype=np.float64).ravel().tofile(handle)

    def load(self, key):
        """Return everything appended under *key* as one flat array."""
        path = self._paths.get(key)
        if path is None:
            return np.empty(0, dtype=np.float64)
        return np.fromfile(path, dtype=np.float64)

    def keys(self):
        """Return the keys that have data stored."""
        return list(self._paths)


def _compute_stats(valid, bins):
    """Return mean/median/std/center-of-mass/n for a 1-D finite array."""
    if valid.size:
        mean = float(np.mean(valid))
        median = float(np.median(valid))
        std = float(np.std(valid))
        counts, edges = np.histogram(valid, bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2.0
        com = (
            float(np.average(centers, weights=counts))
            if counts.sum() > 0
            else float("nan")
        )
    else:
        mean = median = std = com = float("nan")
    return {
        "mean": mean,
        "median": median,
        "std": std,
        "com": com,
        "n": int(valid.size),
    }


def _save_histogram_png(
    values,
    bins,
    path,
    metric,
    colormap_colors=None,
    contrast_limits=None,
    value_range=None,
    dpi=300,
):
    """Save a histogram of ``values`` to ``path`` (matplotlib, no GUI).

    ``value_range`` (``(min, max)``) bounds the histogram to the chosen range
    limits so the image and histogram share the same range.
    """
    valid = np.asarray(values)[np.isfinite(values)]
    if value_range is not None and valid.size:
        low, high = value_range
        valid = valid[(valid >= low) & (valid <= high)]
    if not valid.size:
        return

    hw = HistogramWidget()
    hw.display_mode = "Merged"
    hw.white_background = False
    hw.xlabel = metric
    if colormap_colors is not None or contrast_limits is not None:
        hw.update_colormap(colormap_colors, contrast_limits)
    hw.update_data(valid)

    hw._style_axes(export_mode=True)
    hw.fig.canvas.draw_idle()
    use_transparent = not hw._white_background
    hw.fig.savefig(
        path,
        dpi=dpi,
        bbox_inches='tight',
        transparent=use_transparent,
        facecolor='white' if hw._white_background else 'none',
    )


def _save_histogram_csv(values, bins, path, value_range=None):
    """Save histogram bin centers and counts of ``values`` to a CSV.

    ``value_range`` (``(min, max)``) bounds the histogram to the chosen range
    limits so the CSV matches the range-limited image / PNG histogram.
    """
    valid = np.asarray(values).ravel()
    valid = valid[np.isfinite(valid)]
    if value_range is not None and valid.size:
        low, high = value_range
        valid = valid[(valid >= low) & (valid <= high)]
        counts, edges = np.histogram(valid, bins=bins, range=(low, high))
    else:
        counts, edges = np.histogram(valid, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Bin Center", "Count"])
        for center, count in zip(centers, counts, strict=False):
            writer.writerow([f"{center:.6f}", int(count)])


def _store_plot_settings(layer, plot_settings, group_config=None):
    """Persist phasor-plot display settings into the layer settings.

    These have no effect on the pixel data; they are read back by the plotter
    (``_restore_plot_settings_from_metadata``) when the file is reopened, and
    by the batch widget's *Copy settings* feature. ``group_config`` (the
    multi-layer grouping / histogram display config) is stored too so it can be
    copied back.
    """
    settings = layer.metadata.setdefault("settings", {})
    settings["semi_circle"] = plot_settings["semi_circle"]
    settings["log_scale"] = plot_settings["log_scale"]
    settings["white_background"] = plot_settings["white_background"]
    settings["colormap"] = plot_settings["colormap"]
    if group_config:
        settings["batch_group_config"] = {
            "mode": group_config.get("mode"),
            "assignments": group_config.get("assignments", {}),
            "group_names": {
                str(k): v
                for k, v in group_config.get("group_names", {}).items()
            },
            "group_colors": {
                str(k): v
                for k, v in group_config.get("group_colors", {}).items()
            },
            "show_sd": group_config.get("show_sd"),
            "central_tendency": group_config.get("central_tendency"),
            "show_legend": group_config.get("show_legend"),
        }


def _safe_suffix(name):
    """Turn a layer name into a filesystem-safe filename fragment."""
    keep = []
    for ch in name:
        keep.append(ch if (ch.isalnum() or ch in "-_") else "_")
    return "".join(keep)
