"""Interoperability bridge between napari-phasors and FLIMari.

`FLIMari <https://github.com/GuangchenW/FLIMari>`_ is a separate napari plugin
for phasor-based FLIM analysis. It exposes a lightweight callback bridge,
``flimari.core.bridge.import_from_napari_phasors(data_list)``, that accepts a
list of plain dicts describing pre-computed phasor datasets and builds a
``flimari...ExternalDataset`` from each. This module builds those dicts from
napari-phasors image layers and hands them to a running FLIMari session, live,
within the same napari process -- no files are written.

Recovering total photon counts
-------------------------------
FLIMari filters and thresholds on *total photon counts*, whereas napari-phasors
keeps only the *mean* intensity that ``phasorpy.phasor.phasor_from_signal``
returns after discarding the raw signal. By definition that mean is

.. math:: F_{DC} = \\frac{1}{K} \\sum_{k=0}^{K-1} F_k

i.e. the sum of the signal along the histogram axis divided by the number of
samples ``K``. The per-pixel total photon count is therefore recovered exactly
as ``mean * K`` -- no need to re-read or re-transform the raw data files.

``K`` is the length of the aggregate decay curve that napari-phasors already
stores in ``metadata['summed_signal']`` (``np.sum`` of the signal over the
spatial axes leaves one value per histogram bin). When it is unavailable (for
example, an R64/REF or third-party OME-TIFF whose raw signal was never
stored), ``K`` can instead be recovered from the original raw file via
:func:`histogram_bins_from_raw_file`. If neither is available, only the mean
intensity is sent and FLIMari falls back to using it as a counts proxy.
"""

from __future__ import annotations

import os
from contextlib import suppress
from typing import Any

import numpy as np

#: Max harmonics FLIMari consumes (it requires harmonic ``[1, 2]``).
_MAX_HARMONICS = 2

FLIMARI_INSTALL_HINT = (
    "Could not reach FLIMari. Make sure it is installed "
    "(see https://github.com/GuangchenW/FLIMari)."
)

#: napari plugin name and dock widget display name, as declared in
#: FLIMari's ``napari.yaml`` manifest. Used to open/dock it automatically.
_FLIMARI_PLUGIN_NAME = "flimari"
_FLIMARI_WIDGET_NAME = "Phasor Analysis"


class FlimariNotAvailable(RuntimeError):
    """Raised when the FLIMari bridge module cannot be imported."""


def _import_bridge():
    """Import and return FLIMari's ``core.bridge`` module.

    Raises
    ------
    FlimariNotAvailable
        If FLIMari is not installed / importable.
    """
    try:
        from flimari.core import bridge
    except (
        Exception
    ) as exc:  # noqa: BLE001 - any import failure means "unavailable"
        raise FlimariNotAvailable(FLIMARI_INSTALL_HINT) from exc
    return bridge


def _open_flimari_dock(viewer: Any) -> None:
    """Open and dock FLIMari's main widget in the given napari viewer.

    Constructing FLIMari's widget registers its import callback with the
    bridge synchronously (``SampleManagerWidget.__init__`` calls
    ``register_import_callback``), so once this call returns FLIMari is
    ready to receive data -- no need to wait or poll.

    Raises
    ------
    FlimariNotAvailable
        If the dock widget could not be opened (for example, FLIMari is
        installed as a library but not registered as a napari plugin).
    """
    try:
        viewer.window.add_plugin_dock_widget(
            plugin_name=_FLIMARI_PLUGIN_NAME,
            widget_name=_FLIMARI_WIDGET_NAME,
        )
    except Exception as exc:  # noqa: BLE001
        raise FlimariNotAvailable(
            "Could not open the FLIMari dock widget automatically. "
            "Please open it manually from Plugins > FLIMari."
        ) from exc


def _as_harmonic_stack(array: Any) -> np.ndarray:
    """Return ``array`` with a leading harmonic axis (``_MAX_HARMONICS`` max).

    A 2D ``(Y, X)`` array is promoted to ``(1, Y, X)``; arrays that already
    carry a harmonic axis are returned with at most the first two harmonics,
    matching FLIMari's ``[1, 2]`` requirement.
    """
    arr = np.asarray(array, dtype=float)
    if arr.ndim == 2:
        arr = arr[np.newaxis]
    return arr[:_MAX_HARMONICS]


def _histogram_bins(metadata: dict) -> int | None:
    """Return ``K``, the number of histogram bins, or ``None`` if unknown.

    ``K`` is read from the aggregate decay curve stored in
    ``metadata['summed_signal']`` (or, for layers loaded from a processed file,
    ``metadata['settings']['summed_signal']``). Its length along the last axis
    is the number of histogram bins.
    """
    summed = metadata.get("summed_signal")
    if summed is None:
        summed = metadata.get("settings", {}).get("summed_signal")
    if summed is None:
        return None
    arr = np.asarray(summed)
    if arr.ndim == 0 or arr.size == 0:
        return None
    return int(arr.shape[-1])


def get_layer_histogram_bins(layer: Any) -> int | None:
    """Return the histogram-bin count ``K`` recorded for ``layer``, else None.

    ``None`` means the layer carries no ``summed_signal`` (e.g. it was loaded
    from an R64/REF or third-party OME-TIFF), so photon counts cannot be
    recovered from metadata alone -- the original raw file is needed, see
    :func:`histogram_bins_from_raw_file`.
    """
    metadata = getattr(layer, "metadata", None) or {}
    return _histogram_bins(metadata)


def _means_match(reference: np.ndarray, candidate: np.ndarray) -> bool:
    """Return whether two mean-intensity images are the same up to rounding."""
    if reference.shape != candidate.shape:
        return False
    return bool(
        np.allclose(reference, candidate, rtol=1e-2, atol=1e-3, equal_nan=True)
    )


def histogram_bins_from_raw_file(
    path: str, reference_mean: Any | None = None
) -> int:
    """Recover the histogram-bin count ``K`` from an original raw FLIM file.

    Re-reads ``path`` with napari-phasors' raw reader and returns the number
    of histogram/time bins -- all that is needed to recover total photon
    counts as ``mean * K``. When ``reference_mean`` is provided, the file is
    validated against it (matching image shape and intensity) so that
    assigning the wrong file is caught rather than producing wrong counts.

    Parameters
    ----------
    path : str
        Path to the original raw FLIM file (``.ptu``, ``.fbd``, ``.sdt``, ...).
    reference_mean : array-like, optional
        The layer's mean-intensity image (``metadata['original_mean']``) used
        to confirm the file corresponds to the layer.

    Returns
    -------
    int
        The number of histogram bins ``K``.

    Raises
    ------
    ValueError
        If the file cannot be read, contains no raw histogram data, or does
        not match ``reference_mean``.
    """
    from ._reader import raw_file_reader

    try:
        layers = raw_file_reader(path)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            f"could not read {os.path.basename(path)}: {exc}"
        ) from exc

    k: int | None = None
    means: list[np.ndarray] = []
    for entry in layers or []:
        if not (isinstance(entry, (list, tuple)) and len(entry) > 1):
            continue
        attrs = entry[1] if isinstance(entry[1], dict) else {}
        meta = attrs.get("metadata", {})
        summed = meta.get("summed_signal")
        if summed is not None:
            arr = np.asarray(summed)
            if arr.size:
                # K is identical across channels of a raw file.
                k = int(arr.shape[-1])
        mean = meta.get("original_mean")
        if mean is not None:
            means.append(np.asarray(mean, dtype=float))

    if k is None:
        raise ValueError(
            f"{os.path.basename(path)} does not contain raw histogram data."
        )

    if reference_mean is not None:
        ref = np.asarray(reference_mean, dtype=float)
        if not any(_means_match(ref, mean) for mean in means):
            raise ValueError(
                "the selected file does not match this layer "
                "(different image shape or intensity)."
            )

    return k


def _filter_value(settings: dict, key: str, default: int) -> int:
    """Read a median-filter parameter from ``settings['filter']``."""
    filter_settings = settings.get("filter")
    if (
        isinstance(filter_settings, dict)
        and filter_settings.get(key) is not None
    ):
        return int(filter_settings[key])
    return int(default)


#: Metadata keys a layer must carry to be exportable to FLIMari. Derived
#: layers that hold analysis results rather than phasor coordinates -- e.g.
#: component-analysis fraction images, which only carry
#: ``fraction_data_original`` -- do not have these and are excluded.
_REQUIRED_PHASOR_KEYS = ("G", "S", "G_original", "S_original")


def has_phasor_data(layer: Any) -> bool:
    """Return whether ``layer`` carries the phasor metadata FLIMari needs.

    Used both to decide what to export and to drive the "Export Phasor
    Layer to FLIMARI" button's enabled state: layers produced by other
    analyses (component analysis, FRET, phasor mapping, ...) do not carry
    raw phasor coordinates and must not be offered for export.
    """
    metadata = getattr(layer, "metadata", None) or {}
    return all(
        key in metadata and metadata[key] is not None
        for key in _REQUIRED_PHASOR_KEYS
    )


def build_flimari_dataset(
    layer: Any, histogram_bins: int | None = None
) -> dict | None:
    """Build a FLIMari import dict from a napari-phasors image layer.

    The returned dict follows the schema documented in
    ``flimari.core.bridge`` and additionally carries a ``counts`` key holding
    the recovered per-pixel total photon count (``mean * K``). FLIMari uses
    ``counts`` when present and otherwise falls back to ``mean``.

    Parameters
    ----------
    layer : napari.layers.Image
        Layer carrying napari-phasors phasor metadata (``G``, ``S``,
        ``G_original``, ``S_original``, ``original_mean``, ...).
    histogram_bins : int, optional
        Number of histogram bins ``K`` to use for recovering photon counts.
        Overrides the value derived from the layer metadata; used when ``K``
        was recovered from the original raw file for a layer whose metadata
        lacks ``summed_signal`` (see :func:`histogram_bins_from_raw_file`).

    Returns
    -------
    dict or None
        The import dict, or ``None`` if ``layer`` has no phasor metadata.
    """
    if not has_phasor_data(layer):
        return None

    metadata = layer.metadata
    settings = metadata.get("settings", {}) or {}

    g = _as_harmonic_stack(metadata["G"])
    s = _as_harmonic_stack(metadata["S"])
    g_original = _as_harmonic_stack(metadata["G_original"])
    s_original = _as_harmonic_stack(metadata["S_original"])
    n_harmonics = int(g.shape[0])

    payload: dict[str, Any] = {
        "name": str(getattr(layer, "name", "napari-phasors layer")),
        "channel": _safe_int(settings.get("channel"), 0),
        "g": g,
        "s": s,
        "g_original": g_original,
        "s_original": s_original,
        # napari-phasors bakes any calibration into both G and G_original,
        # so identity coefficients keep FLIMari from applying it a second time.
        "calibration_phase": [0.0] * n_harmonics,
        "calibration_modulation": [1.0] * n_harmonics,
        "filter_size": _filter_value(settings, "size", 3),
        "filter_repeat": _filter_value(settings, "repeat", 0),
    }

    # Only send a frequency when we actually know it; otherwise let FLIMari
    # apply its own default rather than forcing a possibly-wrong value.
    frequency = settings.get("frequency")
    if frequency is not None:
        with suppress(TypeError, ValueError):
            payload["frequency"] = float(frequency)

    mean = metadata.get("original_mean")
    mean_arr = None if mean is None else np.asarray(mean, dtype=float)
    if mean_arr is not None and mean_arr.size:
        payload["mean"] = mean_arr

    k = (
        histogram_bins
        if histogram_bins is not None
        else _histogram_bins(metadata)
    )
    threshold = settings.get("threshold")
    threshold_upper = settings.get("threshold_upper")

    if k is not None and mean_arr is not None and mean_arr.size:
        # Recover true per-pixel photon counts: mean is sum-over-histogram / K.
        payload["counts"] = np.rint(mean_arr * k)
        # napari-phasors thresholds are expressed in mean-intensity units;
        # scale them to photon-count units so they match `counts`.
        payload["min_count"] = int(round(_safe_float(threshold, 0.0) * k))
        payload["max_count"] = (
            int(round(_safe_float(threshold_upper, 0.0) * k))
            if threshold_upper is not None
            else None
        )
    else:
        # No histogram-bin count available: fall back to mean-based thresholds
        # (FLIMari will use `mean` as its counts proxy, as it does today).
        payload["min_count"] = int(round(_safe_float(threshold, 0.0)))
        payload["max_count"] = (
            int(round(_safe_float(threshold_upper, 0.0)))
            if threshold_upper is not None
            else None
        )

    return payload


def send_layers_to_flimari(
    layers: list[Any],
    viewer: Any | None = None,
    histogram_bins: dict[Any, int] | None = None,
) -> tuple[list[dict], list[str], bool]:
    """Send phasor layers to a running FLIMari session via its bridge.

    If FLIMari's dock widget is not open yet, its bridge raises a
    ``RuntimeError``. Rather than surfacing that as an error, when a
    ``viewer`` is provided this opens and docks FLIMari's widget (which
    registers its import callback synchronously) and retries once.

    Parameters
    ----------
    layers : list of napari.layers.Image
        Candidate layers to export. Layers without phasor metadata are skipped.
    viewer : napari.viewer.Viewer, optional
        Viewer used to open FLIMari's dock widget if it is not open yet.
        If not provided, a closed FLIMari dock surfaces as a ``RuntimeError``.
    histogram_bins : dict, optional
        Mapping of layer to its histogram-bin count ``K``, used to recover
        photon counts for layers whose metadata lacks ``summed_signal`` (for
        example, ``K`` recovered from a user-assigned original raw file).

    Returns
    -------
    (sent, skipped, opened_dock) : tuple of (list of dict, list of str, bool)
        The payloads handed to FLIMari, the names of skipped layers, and
        whether FLIMari's dock widget had to be opened automatically.

    Raises
    ------
    FlimariNotAvailable
        If FLIMari is not installed, or its dock could not be opened
        automatically.
    ValueError
        If none of the layers carry phasor data.
    RuntimeError
        Propagated from FLIMari's bridge when no FLIMari dock is open and
        no ``viewer`` was provided to open one automatically.
    """
    bridge = _import_bridge()
    overrides = histogram_bins or {}

    sent: list[dict] = []
    skipped: list[str] = []
    for layer in layers:
        payload = build_flimari_dataset(
            layer, histogram_bins=overrides.get(layer)
        )
        if payload is None:
            skipped.append(str(getattr(layer, "name", "?")))
        else:
            sent.append(payload)

    if not sent:
        raise ValueError(
            "No selected layer contains phasor data to send to FLIMari."
        )

    opened_dock = False
    try:
        bridge.import_from_napari_phasors(sent)
    except RuntimeError:
        if viewer is None:
            raise
        _open_flimari_dock(viewer)
        opened_dock = True
        bridge.import_from_napari_phasors(sent)

    return sent, skipped, opened_dock


def _safe_int(value: Any, default: int) -> int:
    """Best-effort int conversion with a fallback."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float) -> float:
    """Best-effort float conversion with a fallback."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
