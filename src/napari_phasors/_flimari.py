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
example, a processed OME-TIFF from a third party with no summed signal),
counts cannot be recovered and only the mean intensity is sent -- exactly the
behaviour FLIMari falls back to today.
"""

from __future__ import annotations

from contextlib import suppress
from typing import Any

import numpy as np

#: Max harmonics FLIMari consumes (it requires harmonic ``[1, 2]``).
_MAX_HARMONICS = 2

FLIMARI_INSTALL_HINT = (
    "Could not reach FLIMari. Make sure it is installed "
    "(see https://github.com/GuangchenW/FLIMari)."
)


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


def _filter_value(settings: dict, key: str, default: int) -> int:
    """Read a median-filter parameter from ``settings['filter']``."""
    filter_settings = settings.get("filter")
    if (
        isinstance(filter_settings, dict)
        and filter_settings.get(key) is not None
    ):
        return int(filter_settings[key])
    return int(default)


def build_flimari_dataset(layer: Any) -> dict | None:
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

    Returns
    -------
    dict or None
        The import dict, or ``None`` if ``layer`` has no phasor metadata.
    """
    metadata = getattr(layer, "metadata", None) or {}
    required = ("G", "S", "G_original", "S_original")
    if not all(
        key in metadata and metadata[key] is not None for key in required
    ):
        return None

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

    k = _histogram_bins(metadata)
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


def send_layers_to_flimari(layers: list[Any]) -> tuple[list[dict], list[str]]:
    """Send phasor layers to a running FLIMari session via its bridge.

    Parameters
    ----------
    layers : list of napari.layers.Image
        Candidate layers to export. Layers without phasor metadata are skipped.

    Returns
    -------
    (sent, skipped) : tuple of (list of dict, list of str)
        The payloads handed to FLIMari and the names of skipped layers.

    Raises
    ------
    FlimariNotAvailable
        If FLIMari is not installed.
    ValueError
        If none of the layers carry phasor data.
    RuntimeError
        Propagated from FLIMari's bridge when no FLIMari dock is open to
        receive the data.
    """
    bridge = _import_bridge()

    sent: list[dict] = []
    skipped: list[str] = []
    for layer in layers:
        payload = build_flimari_dataset(layer)
        if payload is None:
            skipped.append(str(getattr(layer, "name", "?")))
        else:
            sent.append(payload)

    if not sent:
        raise ValueError(
            "No selected layer contains phasor data to send to FLIMari."
        )

    bridge.import_from_napari_phasors(sent)
    return sent, skipped


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
