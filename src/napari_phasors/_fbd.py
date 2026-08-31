"""Read FLIMbox FBD files with full control over reconstruction settings.

:py:func:`phasorpy.io.signal_from_fbd` calls :py:meth:`fbdfile.FbdFile.asimage`
without forwarding ``refine``. Because ``refine`` defaults to ``True``,
:py:meth:`fbdfile.FbdFile.refine_settings` recomputes ``pixel_dwell_time`` and
``laser_factor`` from the detected frame durations and silently discards any
value passed by the caller.

This module provides a drop-in replacement that forwards ``refine`` and any
other :py:class:`fbdfile.FbdFile` setting (``scanner_line_start``,
``scanner_frame_start``, ``pixel_dwell_time``, ...), so files whose headers do
not describe the acquisition correctly can still be reconstructed.

Files recorded with an IOTech scanner card are a known case: the header's
``x_starting_pixel`` omits the hardware trigger latency that SimFCS applies
internally, and the refined ``laser_factor`` forces exactly ``frame_size``
lines per frame, which shears the image along the slow scan axis. See
:py:func:`iotech_laser_factor`.

Finding those settings by hand is tedious, so :py:func:`match_reference_settings`
derives them from a SimFCS reference image (an R64/REF file exported for the
same acquisition) by maximizing the correlation between the reconstruction and
the reference.

"""

from __future__ import annotations

__all__ = [
    "DEFAULT_CANDIDATES",
    "IOTECH",
    "REFERENCE_EXTENSIONS",
    "FbdReconstructionSettings",
    "find_reference_file",
    "image_correlation",
    "iotech_laser_factor",
    "match_reference_settings",
    "read_reference_image",
    "signal_from_fbd",
]

import math
import os
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence
    from os import PathLike

    from numpy.typing import ArrayLike, NDArray
    from xarray import DataArray

_UNSET = object()
"""Sentinel marking an argument the caller did not provide."""

IOTECH = "iotech"
"""Pass as ``laser_factor`` to derive it with :py:func:`iotech_laser_factor`.

The numeric value differs between :py:mod:`fbdfile` releases, so prefer this
over hard-coding a number.
"""

REFERENCE_EXTENSIONS = (".r64", ".ref")
"""Extensions of SimFCS referenced files usable as a reconstruction reference."""

DEFAULT_CANDIDATES: tuple[tuple[float | str, bool | None], ...] = (
    (IOTECH, False),
    (-1.0, False),
    (-1.0, True),
)
"""``(laser_factor, refine)`` pairs tried by :py:func:`match_reference_settings`.

In order: the SimFCS-matching factor derived from the header, the header's
own factor used verbatim, and the factor refined from the detected frame
durations (what :py:func:`phasorpy.io.signal_from_fbd` always does).
"""


class FbdReconstructionSettings(NamedTuple):
    """Reconstruction settings matching a SimFCS reference image.

    Returned by :py:func:`match_reference_settings`.

    Attributes
    ----------
    scanner_line_start : int
        First valid pixel of the scan line.
    laser_factor : float or str
        Value to pass to :py:func:`signal_from_fbd`. Either a number, or
        :py:data:`IOTECH` when the factor is derived from the header.
    refine : bool or None
        Value to pass to :py:func:`signal_from_fbd`.
    correlation : float
        Pearson correlation between the reconstruction and the reference.
    laser_factor_value : float
        Numeric factor the winning reconstruction actually used. Reported
        for display only; passing it back with ``refine=True`` does not
        reproduce the reconstruction, because refining also replaces
        ``pixel_dwell_time``.

    """

    scanner_line_start: int
    laser_factor: float | str
    refine: bool | None
    correlation: float
    laser_factor_value: float

    def as_reader_options(self) -> dict[str, Any]:
        """Return the settings as keyword arguments for the FBD reader."""
        return {
            "laser_factor": self.laser_factor,
            "scanner_line_start": self.scanner_line_start,
            "refine": self.refine,
        }


def iotech_laser_factor(fbd: Any, /) -> float:
    """Return ``laser_factor`` matching SimFCS for an IOTech scanner file.

    SimFCS applies its dwell-time correction at the full phase resolution of
    the FLIMbox, ``pmax * harmonics``, whereas :py:mod:`fbdfile` applies it at
    ``pmax``. For second-harmonic files the two differ by about 4e-3, enough
    to shift the reconstructed image by tens of pixels between the first and
    last line of a frame.

    The returned factor targets the FLIMbox units per scanner sample that
    SimFCS uses::

        line_time / line_length * 1e-6 * P / (P - 1)
            * laser_frequency * header laser_factor

    with ``P = pmax * harmonics``. It is expressed relative to
    ``fbd.pixel_dwell_time``, so it stays correct across :py:mod:`fbdfile`
    versions: 2026.3.20 reports a dwell time of 31.875 us for these files
    where 2026.6.6 reports 32.0, and the factor absorbs the difference.

    Parameters
    ----------
    fbd : fbdfile.FbdFile
        Open FBD file. Must have a binary header.

    Returns
    -------
    float
        Value to pass as ``laser_factor``.

    Raises
    ------
    ValueError
        If the file has no binary header to read ``laser_factor`` from, if
        its phase resolution is too small to correct, or if its dwell time
        is not positive.

    Examples
    --------
    >>> from fbdfile import FbdFile
    >>> with FbdFile(filename) as fbd:  # doctest: +SKIP
    ...     factor = iotech_laser_factor(fbd)
    ...

    """
    if fbd.header is None:
        msg = "FBD file has no binary header"
        raise ValueError(msg)
    header_factor = float(fbd.header["laser_factor"])
    pmax = int(fbd.pmax)
    phase_max = pmax * int(fbd.harmonics)
    if pmax < 2 or phase_max < 2:
        msg = f"FBD file has too few phase bins: {pmax=}, {phase_max=}"
        raise ValueError(msg)

    dwell_time = float(fbd.pixel_dwell_time)
    nominal_dwell_time = _nominal_dwell_time(fbd, dwell_time)
    if dwell_time <= 0.0 or nominal_dwell_time <= 0.0:
        msg = f"FBD file has a non-positive dwell time: {dwell_time}"
        raise ValueError(msg)

    return (
        header_factor
        * (nominal_dwell_time / dwell_time)
        * (phase_max / (phase_max - 1))
        / (pmax / (pmax - 1))
    )


def _nominal_dwell_time(fbd: Any, default: float, /) -> float:
    """Return the dwell time the scanner was configured with.

    Derived from the header's line timing rather than from
    :py:mod:`fbdfile`'s dwell-time table, which has carried different values
    for the same index across releases. Falls back to `default` when the
    header does not describe the line timing.
    """
    try:
        line_time = float(fbd.header["line_time"])
        line_length = int(fbd.header["line_length"])
    except (KeyError, IndexError, TypeError, ValueError):
        return default
    if line_time <= 0.0 or line_length <= 0:
        return default
    return line_time / line_length


def _is_iotech(laser_factor: float | str, /) -> bool:
    """Return whether `laser_factor` requests the derived IOTech factor."""
    if not isinstance(laser_factor, str):
        return False
    if laser_factor != IOTECH:
        msg = f"{laser_factor=!r} is not a number or {IOTECH!r}"
        raise ValueError(msg)
    return True


def signal_from_fbd(
    filename: str | PathLike[Any],
    /,
    *,
    frame: int | None = None,
    channel: int | None = 0,
    keepdims: bool = False,
    laser_factor: float | str = -1.0,
    refine: Any = _UNSET,
    **kwargs: Any,
) -> DataArray:
    """Return phase histogram and metadata from FLIMbox FBD file.

    Same signature and return value as
    :py:func:`phasorpy.io.signal_from_fbd`, plus control over ``refine`` and
    the remaining :py:class:`fbdfile.FbdFile` settings.

    Parameters
    ----------
    filename : str or Path
        Name of FLIMbox FBD file to read.
    frame : int, optional
        Index of frame to return.
        By default, return all frames.
        If < 0, integrate time axis, else return specified frame.
    channel : int or None, optional
        Index of channel to return.
        By default, return the first channel.
        If None, return all channels.
    keepdims : bool, optional, default: False
        Return reduced axes as length-1 dimensions.
    laser_factor : float or str, optional, default: -1
        Factor to correct dwell_time / laser_frequency.
        If < 0, use the value stored in the file.
        If ``'iotech'`` (:py:data:`IOTECH`), derive the factor that matches
        SimFCS with :py:func:`iotech_laser_factor`. Prefer this over a
        literal, which is only valid for one :py:mod:`fbdfile` release.
    refine : bool or None, optional
        Refine ``pixel_dwell_time`` and ``laser_factor`` from the detected
        frame durations: True=always, None=if needed, False=never.
        By default, refine unless an explicit `laser_factor` was passed,
        because refining overwrites it.
    **kwargs
        Optional arguments passed to :py:class:`fbdfile.FbdFile`, such as
        ``scanner_line_start``, ``scanner_frame_start``, ``frame_size``,
        ``pixel_dwell_time``, or ``laser_frequency``.

    Returns
    -------
    xarray.DataArray
        Phase histogram with axes codes ``'TCYXH'`` and type `uint16`, and
        selected metadata:

        - ``coords['H']``: cross-correlation phases in radians.
        - ``attrs['frequency']``: repetition frequency in MHz.
        - ``attrs['harmonic']``: harmonic contained in phase histogram.
        - ``attrs['laser_factor']``: factor used for reconstruction.
        - ``attrs['scanner_line_start']``: first valid pixel of scan line.
        - ``attrs['flimbox_header']``: FBD binary header, if any.
        - ``attrs['flimbox_firmware']``: FLIMbox firmware settings, if any.
        - ``attrs['flimbox_settings']``: Settings from FBS XML, if any.

    Raises
    ------
    ValueError
        If file is not a FLIMbox FBD file.
    IndexError
        If frame or channel index is out of bounds.

    Examples
    --------
    Reconstruct an IOTech scanner file the way SimFCS does:

    >>> signal = signal_from_fbd(  # doctest: +SKIP
    ...     filename, laser_factor=IOTECH, scanner_line_start=51
    ... )

    """
    import fbdfile
    from xarray import DataArray

    want_iotech = _is_iotech(laser_factor)
    if refine is _UNSET:
        # refining overwrites laser_factor, so honor an explicit value
        refine = not (want_iotech or float(laser_factor) >= 0.0)

    integrate_frames = 0 if frame is None or frame >= 0 else 1

    with fbdfile.FbdFile(
        filename,
        laser_factor=-1.0 if want_iotech else float(laser_factor),
        **kwargs,
    ) as fbd:
        if want_iotech:
            # after opening, so the header is available to derive it from
            fbd.laser_factor = iotech_laser_factor(fbd)
        data = fbd.asimage(integrate_frames=integrate_frames, refine=refine)
        if integrate_frames:
            frame = None
        copy = False
        axes = "TCYXH"
        if channel is None:
            if not keepdims and data.shape[1] == 1:
                data = data[:, 0]
                axes = "TYXH"
        else:
            if channel < 0 or channel >= data.shape[1]:
                msg = f"{channel=} is out of bounds [0, {data.shape[1] - 1}]"
                raise IndexError(msg)
            if keepdims:
                data = data[:, channel : channel + 1]
            else:
                data = data[:, channel]
                axes = "TYXH"
            copy = True
        if frame is None:
            if not keepdims and data.shape[0] == 1:
                data = data[0]
                axes = axes[1:]
        else:
            if frame < 0 or frame >= data.shape[0]:
                msg = f"{frame=} is out of bounds [0, {data.shape[0] - 1}]"
                raise IndexError(msg)
            if keepdims:
                data = data[frame : frame + 1]
            else:
                data = data[frame]
                axes = axes[1:]
            copy = True
        if copy:
            data = data.copy()

        phases = numpy.linspace(
            0.0, numpy.pi * 2, data.shape[-1], endpoint=False
        )
        attrs: dict[str, Any] = {
            "frequency": fbd.laser_frequency * 1e-6,
            "harmonic": fbd.harmonics,
            "laser_factor": fbd.laser_factor,
            "scanner_line_start": fbd.scanner_line_start,
        }
        if fbd.header is not None:
            attrs["flimbox_header"] = fbd.header
        if fbd.fbf is not None:
            attrs["flimbox_firmware"] = fbd.fbf
        if fbd.fbs is not None:
            attrs["flimbox_settings"] = fbd.fbs

    return DataArray(data, dims=tuple(axes), coords={"H": phases}, attrs=attrs)


def find_reference_file(filename: str | PathLike[Any], /) -> str | None:
    """Return the SimFCS reference file recorded next to an FBD file.

    SimFCS writes one referenced file per detector channel, named after the
    FBD file. The ``_ch2_`` companions hold median filtered data and are not
    a faithful reference, so a first-channel file is preferred.

    Parameters
    ----------
    filename : str or Path
        Name of FLIMbox FBD file.

    Returns
    -------
    str or None
        Path of the companion reference file, or None if there is none.

    """
    filename = os.fspath(filename)
    directory = os.path.dirname(os.path.abspath(filename))
    stem = os.path.splitext(os.path.basename(filename))[0]
    if not stem or not os.path.isdir(directory):
        return None

    candidates = [
        os.path.join(directory, name)
        for name in sorted(os.listdir(directory))
        if name.startswith(stem)
        and os.path.splitext(name)[1].lower() in REFERENCE_EXTENSIONS
    ]
    if not candidates:
        return None
    first_channel = [
        path
        for path in candidates
        if "_ch1_" in os.path.basename(path).lower()
    ]
    return (first_channel or candidates)[0]


def read_reference_image(
    filename: str | PathLike[Any], /
) -> NDArray[numpy.float64]:
    """Return the mean intensity image from a SimFCS referenced file.

    Parameters
    ----------
    filename : str or Path
        Name of SimFCS R64 or REF file.

    Returns
    -------
    numpy.ndarray
        Two-dimensional mean intensity image, with NaN replaced by zero.

    Raises
    ------
    ValueError
        If the file does not contain a two-dimensional image.

    """
    from phasorpy.io import phasor_from_simfcs_referenced

    mean = phasor_from_simfcs_referenced(filename)[0]
    image = numpy.nan_to_num(numpy.asarray(mean, dtype=numpy.float64))
    if image.ndim != 2:
        msg = (
            f"reference file contains a {image.ndim}-dimensional image, "
            "expected a single two-dimensional image"
        )
        raise ValueError(msg)
    return image


def image_correlation(a: ArrayLike, b: ArrayLike, /) -> float:
    """Return the Pearson correlation between two images of equal shape.

    Returns NaN when either image is flat, so callers comparing scores can
    treat it as "no worse than anything".
    """
    x = numpy.asarray(a, dtype=numpy.float64)
    y = numpy.asarray(b, dtype=numpy.float64)
    x = x - x.mean()
    y = y - y.mean()
    denominator = math.sqrt(float((x * x).sum()) * float((y * y).sum()))
    if denominator == 0.0:
        return float("nan")
    return float((x * y).sum() / denominator)


def match_reference_settings(
    filename: str | PathLike[Any],
    reference: str | PathLike[Any] | ArrayLike,
    /,
    *,
    channel: int | None = 0,
    frame: int | None = -1,
    line_start_range: tuple[int, int] | None = None,
    candidates: Iterable[tuple[float | str, bool | None]] | None = None,
    progress: Callable[[int, int], Any] | None = None,
    **kwargs: Any,
) -> FbdReconstructionSettings:
    """Return the reconstruction settings best matching a reference image.

    Each ``(laser_factor, refine)`` candidate is decoded once over the *full*
    scanner frame, and every ``scanner_line_start`` is then scored by sliding
    the reference across that decoded frame. This is what
    :py:meth:`fbdfile.FbdFile.asimage` does internally when
    ``square_frame=True``, so the scan costs one decode per candidate instead
    of one decode per line start.

    Parameters
    ----------
    filename : str or Path
        Name of FLIMbox FBD file to reconstruct.
    reference : str or Path or array_like
        SimFCS R64/REF file recorded for the same acquisition, or a
        two-dimensional intensity image to match.
    channel : int or None, optional, default: 0
        Index of channel to match. If None, sum all channels.
    frame : int or None, optional, default: -1
        Index of frame to match. If < 0, integrate all frames.
    line_start_range : tuple of int, optional
        Inclusive ``(low, high)`` bounds for ``scanner_line_start``.
        By default, try every position the reference fits in.
    candidates : iterable of tuple, optional
        ``(laser_factor, refine)`` pairs to try.
        By default, :py:data:`DEFAULT_CANDIDATES`.
    progress : callable, optional
        Called as ``progress(completed, total)`` before each candidate and
        once after the last one.
    **kwargs
        Optional arguments passed to :py:class:`fbdfile.FbdFile`.

    Returns
    -------
    FbdReconstructionSettings
        Settings of the highest scoring reconstruction.

    Raises
    ------
    ValueError
        If `reference` is not a two-dimensional image, or if none of the
        candidates could be evaluated.

    Examples
    --------
    >>> settings = match_reference_settings(  # doctest: +SKIP
    ...     filename, find_reference_file(filename)
    ... )
    >>> signal = signal_from_fbd(  # doctest: +SKIP
    ...     filename, **settings.as_reader_options()
    ... )

    """
    if isinstance(reference, (str, os.PathLike)):
        image = read_reference_image(reference)
    else:
        image = numpy.nan_to_num(numpy.asarray(reference, dtype=numpy.float64))
    if image.ndim != 2:
        msg = (
            f"reference is {image.ndim}-dimensional, "
            "expected a two-dimensional image"
        )
        raise ValueError(msg)

    trials = tuple(DEFAULT_CANDIDATES if candidates is None else candidates)
    if not trials:
        msg = "no candidate reconstruction settings to try"
        raise ValueError(msg)

    best: FbdReconstructionSettings | None = None
    failures: list[str] = []
    for index, (laser_factor, refine) in enumerate(trials):
        if progress is not None:
            progress(index, len(trials))
        try:
            result = _match_candidate(
                filename,
                image,
                laser_factor,
                refine,
                channel,
                frame,
                line_start_range,
                kwargs,
            )
        except Exception as exc:  # noqa: BLE001
            failures.append(f"laser_factor={laser_factor!r}, {refine=}: {exc}")
            continue
        if best is None or result.correlation > best.correlation:
            best = result
    if progress is not None:
        progress(len(trials), len(trials))

    if best is None:
        msg = "no candidate settings could be evaluated: " + "; ".join(
            failures
        )
        raise ValueError(msg)
    return best


def _match_candidate(
    filename: str | PathLike[Any],
    reference: NDArray[numpy.float64],
    laser_factor: float | str,
    refine: bool | None,
    channel: int | None,
    frame: int | None,
    line_start_range: tuple[int, int] | None,
    kwargs: dict[str, Any],
    /,
) -> FbdReconstructionSettings:
    """Return the best ``scanner_line_start`` for one candidate."""
    import fbdfile

    want_iotech = _is_iotech(laser_factor)
    integrate_frames = 0 if frame is None or frame >= 0 else 1

    with fbdfile.FbdFile(
        filename,
        laser_factor=-1.0 if want_iotech else float(laser_factor),
        **kwargs,
    ) as fbd:
        if want_iotech:
            fbd.laser_factor = iotech_laser_factor(fbd)
        data = fbd.asimage(
            integrate_frames=integrate_frames,
            square_frame=False,
            refine=refine,
        )
        frame_size = int(fbd.frame_size)
        # read after asimage: refining replaces it while detecting frames
        laser_factor_value = float(fbd.laser_factor)

    image = _intensity_image(data, frame, channel, integrate_frames)
    start, score = _best_line_start(
        image, reference, frame_size, line_start_range
    )
    return FbdReconstructionSettings(
        scanner_line_start=start,
        laser_factor=IOTECH if want_iotech else float(laser_factor),
        refine=refine,
        correlation=score,
        laser_factor_value=laser_factor_value,
    )


def _intensity_image(
    data: NDArray[Any],
    frame: int | None,
    channel: int | None,
    integrate_frames: int,
    /,
) -> NDArray[numpy.float64]:
    """Return the ``(Y, X)`` intensity image of one frame and channel."""
    index = 0 if integrate_frames else (frame or 0)
    if index < 0 or index >= data.shape[0]:
        msg = f"{frame=} is out of bounds [0, {data.shape[0] - 1}]"
        raise IndexError(msg)
    selected = data[index]
    if channel is None:
        return selected.sum(axis=(0, -1), dtype=numpy.float64)
    if channel < 0 or channel >= selected.shape[0]:
        msg = f"{channel=} is out of bounds [0, {selected.shape[0] - 1}]"
        raise IndexError(msg)
    return selected[channel].sum(axis=-1, dtype=numpy.float64)


def _best_line_start(
    image: NDArray[numpy.float64],
    reference: NDArray[numpy.float64],
    frame_size: int,
    line_start_range: Sequence[int] | None,
    /,
) -> tuple[int, float]:
    """Return the ``(line_start, correlation)`` best matching `reference`."""
    width = reference.shape[1]
    if width != frame_size:
        msg = (
            f"reference is {width} pixels wide but the file reconstructs "
            f"{frame_size} pixel wide frames"
        )
        raise ValueError(msg)
    if width > image.shape[1]:
        msg = (
            f"reference is {width} pixels wide, wider than the "
            f"{image.shape[1]} scanner samples per line"
        )
        raise ValueError(msg)

    height = min(frame_size, image.shape[0], reference.shape[0])
    if height < 1:
        msg = "reconstruction and reference have no rows in common"
        raise ValueError(msg)
    window = reference[:height]

    low = 0
    high = image.shape[1] - width
    if line_start_range is not None:
        low = max(low, int(line_start_range[0]))
        high = min(high, int(line_start_range[1]))
    if low > high:
        msg = f"line_start_range {tuple(line_start_range)!r} is empty"
        raise ValueError(msg)

    best_start = low
    best_score = -math.inf
    for start in range(low, high + 1):
        score = image_correlation(
            image[:height, start : start + width], window
        )
        if score > best_score:
            best_start = start
            best_score = score
    if not math.isfinite(best_score):
        msg = "reconstruction could not be correlated with the reference"
        raise ValueError(msg)
    return best_start, best_score
