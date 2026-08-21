"""
This module contains functions to read files supported by `phasorpy.io`
and computes phasor coordinates with `phasorpy.phasor.phasor_from_signal`

"""

import html
import inspect
import itertools
import json
import os
import threading
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import replace
from typing import Any, Union

import numpy as np
import phasorpy.io as io
import tifffile
import xarray as xr
from napari.utils.colormaps.colormap_utils import CYMRGB, MAGENTA_GREEN
from napari.utils.notifications import show_error
from phasorpy.phasor import phasor_from_signal

from ._parallel import parallel_map, workers_for_memory
from ._stitching import as_tile_sources, blend_phasor_tiles
from ._utils import show_activity_progress

extension_mapping = {
    "raw": {
        ".ptu": lambda path, reader_options: _parse_and_call_io_function(
            path,
            io.signal_from_ptu,
            {"frame": (-1, False), "keepdims": (False, False)},
            reader_options,
        ),
        ".fbd": lambda path, reader_options: _parse_and_call_io_function(
            path,
            io.signal_from_fbd,
            {
                "frame": (-1, False),
                "keepdims": (False, False),
                "channel": (None, False),
            },
            reader_options,
        ),
        ".sdt": lambda path, reader_options: _parse_and_call_io_function(
            path,
            io.signal_from_sdt,
            {},
            reader_options,
        ),
        ".lsm": lambda path, reader_options: _parse_and_call_io_function(
            path,
            io.signal_from_lsm,
            {},
            reader_options,
        ),
        ".tif": lambda path, reader_options: _parse_and_call_io_function(
            path,
            tifffile.imread,
            {},
            reader_options,
        ),
        ".tiff": lambda path, reader_options: _parse_and_call_io_function(
            path,
            tifffile.imread,
            {},
            reader_options,
        ),
        ".czi": lambda path, reader_options: _parse_and_call_io_function(
            path, io.signal_from_czi, {}, reader_options
        ),
        ".flif": lambda path, reader_options: _parse_and_call_io_function(
            path, io.signal_from_flif, {}, reader_options
        ),
        ".bh": lambda path, reader_options: _parse_and_call_io_function(
            path, io.signal_from_bh, {}, reader_options
        ),
        ".b&h": lambda path, reader_options: _parse_and_call_io_function(
            path, io.signal_from_bh, {}, reader_options
        ),
        ".bhz": lambda path, reader_options: _parse_and_call_io_function(
            path, io.signal_from_bhz, {}, reader_options
        ),
        ".lif": lambda path, reader_options: _parse_and_call_io_function(
            path,
            io.signal_from_lif,
            {"image": (None, False), "dim": ("λ", False)},
            reader_options,
        ),
        ".bin": lambda path, reader_options: _parse_and_call_io_function(
            path, io.signal_from_pqbin, {}, reader_options
        ),
        ".json": lambda path, reader_options: _parse_and_call_io_function(
            path,
            io.signal_from_flimlabs_json,
            {"channel": (0, False), "dtype": (None, False)},
            reader_options,
        ),
    },
    "processed": {
        ".ome.tif": lambda path, reader_options: _parse_and_call_io_function(
            path, io.phasor_from_ometiff, {}, reader_options
        ),
        ".ome.tiff": lambda path, reader_options: _parse_and_call_io_function(
            path, io.phasor_from_ometiff, {}, reader_options
        ),
        ".r64": lambda path, reader_options: _parse_and_call_io_function(
            path, io.phasor_from_simfcs_referenced, {}, reader_options
        ),
        ".ref": lambda path, reader_options: _parse_and_call_io_function(
            path, io.phasor_from_simfcs_referenced, {}, reader_options
        ),
        ".ifli": lambda path, reader_options: _parse_and_call_io_function(
            path, io.phasor_from_ifli, {"channel": (0, False)}, reader_options
        ),
        ".lif": lambda path, reader_options: _parse_and_call_io_function(
            path, io.phasor_from_lif, {"image": (None, False)}, reader_options
        ),
        ".json": lambda path, reader_options: _parse_and_call_io_function(
            path,
            io.phasor_from_flimlabs_json,
            {"channel": (0, False)},
            reader_options,
        ),
    },
}
"""This dictionary contains the mapping for reader functions from
`phasorpy.io` supported formats.

Commented file extensions are not supported at the moment.

"""

iter_index_mapping = {
    ".ptu": "C",
    ".fbd": "C",
    ".lsm": None,
    ".tif": None,
    ".tiff": None,
    '.sdt': "C",
    ".czi": None,
    ".flif": None,
    ".bh": None,
    ".b&h": None,
    ".bhz": None,
    ".lif": None,
    ".bin": None,
    ".json": "C",
}
"""This dictionary contains the mapping for the axis to iterate over
when calculating phasor coordinates in the file.
"""


def napari_get_reader(
    path: str | list[str],
    reader_options: dict | None = None,
    harmonics: Union[int, Sequence[int], None] = None,
) -> Callable | None:
    """Initial reader function to map file extension to
    specific reader functions.

    Parameters
    ----------
    path : str or list of str
        Path to a file, or a list of file paths selected in napari.
    reader_options : dict, optional
        Dictionary containing the arguments to pass to the function.
    harmonics : Union[int, Sequence[int], None], optional
        Harmonic(s) to be processed. Can be a single integer, a sequence of
        integers, or None. Default is None.

    Returns
    -------
    layer_data : list of tuples, or None
        A list of LayerData tuples where each tuple in the list contains a
        napari.layers.Labels layer a tuple  (data, kwargs), where data is
        the mean intensity image as an array, and kwargs is a a dict of
        keyword arguments for the corresponding viewer.add_* method in napari,
        which contains the 'name' of the layer as well as the 'metadata',
        which is also a dict. The values for key 'phasor_features_labels_layer'
        in 'metadata' contain phasor coordinates as columns 'G' and 'S'.

    """
    if isinstance(path, list):
        if len(path) == 0:
            show_error("No files selected.")
            return None

        # Napari may pass a list of paths when selecting multiple files.
        if len(path) > 1:
            extensions = {_get_filename_extension(p)[1] for p in path}
            if len(extensions) != 1:
                show_error(
                    f"All files must share the same extension, got: {extensions}"
                )
                return None

            file_extension = next(iter(extensions))
            if file_extension in extension_mapping["raw"]:
                return lambda paths: raw_file_stack_reader(
                    paths,
                    reader_options=reader_options,
                    harmonics=harmonics,
                )

            show_error(
                "Multi-file loading is only supported for raw file formats."
            )
            return None

        path = path[0]

    extensions_both = set(extension_mapping["raw"].keys()).intersection(
        extension_mapping["processed"].keys()
    )
    path_lower = path.lower()
    if path_lower.endswith(tuple(extensions_both)):
        return lambda path: ambiguous_file_reader(
            path, reader_options=reader_options, harmonics=harmonics
        )
    elif path_lower.endswith(tuple(extension_mapping["processed"].keys())):
        return lambda path: processed_file_reader(
            path, reader_options=reader_options, harmonics=harmonics
        )
    elif path_lower.endswith(tuple(extension_mapping["raw"].keys())):
        return lambda path: raw_file_reader(
            path, reader_options=reader_options, harmonics=harmonics
        )
    else:
        show_error("File extension not supported.")
        return None


def _clamp_harmonics(
    harmonics: Union[int, Sequence[int], None], n_samples: int
):
    """Clamp or convert the requested harmonics to what's valid for n_samples.

    The maximum usable harmonic is `n_samples // 2`. If `harmonics` is None,
    it defaults to `[1, 2]`. If the requested harmonics exceed the maximum,
    they are clamped down. If no valid harmonics remain, a ValueError is
    raised.
    """
    max_h = n_samples // 2
    if max_h < 1:
        raise ValueError(
            f"Not enough samples ({n_samples}) to compute phasor harmonics; need at least 2 samples."
        )

    # Default harmonics
    if harmonics is None:
        harmonics = [1, 2]

    # 'all' means 1..max_h
    if harmonics == "all":
        return list(range(1, max_h + 1))

    # Single int -> list
    if isinstance(harmonics, int):
        h = min(harmonics, max_h)
        return [h]

    # Iterable of ints
    out = []
    for h in harmonics:
        try:
            hi = int(h)
        except (TypeError, ValueError):
            continue
        if hi <= 0:
            continue
        out.append(min(hi, max_h))

    # Remove duplicates while preserving order
    seen = set()
    res = []
    for v in out:
        if v not in seen:
            seen.add(v)
            res.append(v)

    if not res:
        raise ValueError(
            f"No valid harmonics remain for {n_samples} samples. "
            f"Requested harmonics: {harmonics}."
        )
    return res


def ambiguous_file_reader(
    path: str,
    reader_options: dict | None = None,
    harmonics: Union[int, Sequence[int], None] = None,
) -> list[tuple]:
    """Fallback reader that attempts to parse an ambiguous file extension as raw, then processed."""
    try:
        return raw_file_reader(
            path, reader_options=reader_options, harmonics=harmonics
        )
    except Exception as e_raw:  # noqa: BLE001
        try:
            return processed_file_reader(
                path, reader_options=reader_options, harmonics=harmonics
            )
        except Exception as e_processed:  # noqa: BLE001
            raise RuntimeError(
                "Failed to read ambiguous file with both raw and "
                "processed readers. "
                f"raw_file_reader error: {e_raw!r}; "
                f"processed_file_reader error: {e_processed!r}"
            ) from e_processed


def raw_file_reader(
    path: str,
    reader_options: dict | None = None,
    harmonics: Union[int, Sequence[int], None] = None,
) -> list[tuple]:
    """Read raw data files from supported file formats and apply the phasor
    transformation to get mean intensity image and phasor coordinates.

    Parameters
    ----------
    path : str
        Path to file.
    reader_options : dict, optional
        Dictionary containing the arguments to pass to the function.
    harmonics : Union[int, Sequence[int], None], optional
        Harmonic(s) to be processed. Can be a single integer, a sequence of
        integers, or None. Default is None, which sets the first two harmonics
        to be processed.

    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples where each tuple in the list contains a
        napari.layers.Labels layer a tuple  (data, kwargs), where data is
        the mean intensity image as an array, and kwargs is a a dict of
        keyword arguments for the corresponding viewer.add_* method in napari,
        which contains the 'name' of the layer as well as the 'metadata',
        which is also a dict. The values for key 'phasor_features_labels_layer'
        in 'metadata' contain phasor coordinates as columns 'G' and 'S'.

    """
    # Set default harmonics if None is passed
    if harmonics is None:
        harmonics = [1, 2]

    axis_override, keep_signal, filtered_reader_options = (
        _split_widget_reader_options(reader_options)
    )
    filename, file_extension = _get_filename_extension(path)
    raw_data = load_raw_signal(path, filtered_reader_options)

    return _phasor_layers_from_signal(
        raw_data,
        filename=filename,
        file_extension=file_extension,
        harmonics=harmonics,
        axis_override=axis_override,
        keep_signal=keep_signal,
    )


def _split_widget_reader_options(reader_options):
    """Separate widget-only options from those meant for the IO functions.

    Returns ``(axis_override, keep_signal, io_options)``.
    """
    # Extract phasor_axis from reader_options (widget-level parameter)
    # This should not be passed to IO functions
    axis_override = None
    filtered_reader_options = reader_options.copy() if reader_options else {}
    if 'phasor_axis' in filtered_reader_options:
        try:
            axis_override = int(filtered_reader_options.pop('phasor_axis'))
        except (KeyError, ValueError, TypeError):
            filtered_reader_options.pop('phasor_axis', None)

    # Opt-in flag (widget-level, never passed to IO functions): when set, keep
    # the full per-pixel signal and its histogram/spectral axis in the layer
    # metadata so callers (e.g. batch signal export) can average the signal
    # over a masked region. This is memory-heavy, so it is off by default.
    keep_signal = bool(filtered_reader_options.pop('_keep_signal', False))

    # Spatial binning is applied by the mosaic reader, never by the IO
    # functions, so drop it here whichever path the file takes.
    filtered_reader_options.pop('binning', None)
    return axis_override, keep_signal, filtered_reader_options


def load_raw_signal(path, io_options=None):
    """Read the raw signal of a file without computing phasor coordinates.

    Parameters
    ----------
    path : str
        Path to a file in one of the supported raw formats.
    io_options : dict, optional
        Arguments forwarded to the format's ``phasorpy.io`` function. Must
        contain only IO arguments; see :func:`_split_widget_reader_options`.

    Returns
    -------
    xarray.DataArray or numpy.ndarray
        The signal as returned by the format's reader.
    """
    _, file_extension = _get_filename_extension(path)
    io_options = io_options or {}

    # A CZI mosaic's nominal extent covers the whole scanned area, so reading
    # it as one image means allocating an array far larger than the file --
    # terabytes for a slide scan. Refuse it here so every caller gets a clear
    # error instead of exhausting memory.
    mosaic = czi_mosaic_info(path)
    if mosaic is not None:
        height, width = mosaic["canvas_shape"]
        raise ValueError(
            f"{os.path.basename(path)} is a mosaic of {mosaic['n_tiles']} "
            f"tiles spanning {height} x {width} pixels, which cannot be read "
            "as a single image. Import it with 'Open tiled mosaic'."
        )

    # Read SDT multi-file special case
    if file_extension == ".sdt":
        i = 0
        raw_list = []
        while True:
            try:
                _data = extension_mapping["raw"][".sdt"](path, {"index": i})
                raw_list.append(_data)
                i += 1
            except IndexError:
                break
        for _d in raw_list:
            assert (
                _d.shape == raw_list[0].shape
            ), "Shapes from files in .sdt do not match!"
        return xr.concat(raw_list, dim="C")

    return extension_mapping["raw"][file_extension](path, io_options)


def _phasor_layers_from_signal(
    raw_data,
    *,
    filename,
    file_extension,
    harmonics,
    axis_override=None,
    keep_signal=False,
    progress_description=None,
):
    """Compute phasor coordinates for an already-loaded signal.

    Split out of :func:`raw_file_reader` so that a file holding several tiles
    can be read once and then transformed one tile at a time.

    Parameters
    ----------
    raw_data : xarray.DataArray or numpy.ndarray
        Signal returned by :func:`load_raw_signal`, or one slice of it.
    filename : str
        Name used for the resulting layers.
    file_extension : str
        Extension the signal was read from, which selects the channel axis.
    harmonics : int or sequence of int
        Harmonics to compute.
    axis_override : int, optional
        Index of the histogram or spectral axis. Detected when ``None``.
    keep_signal : bool, optional
        Keep the full signal in the layer metadata.
    progress_description : str, optional
        Text shown on the progress bar. Defaults to the file name.

    Returns
    -------
    list of tuple
        Napari layer-data tuples, one per channel.
    """
    settings = {}
    if (
        file_extension != '.fbd'
        and hasattr(raw_data, "attrs")
        and 'frequency' in raw_data.attrs
    ):
        settings['frequency'] = raw_data.attrs['frequency']

    layers = []
    iter_axis = iter_index_mapping[file_extension]
    has_dims = hasattr(raw_data, 'dims')
    raw_dims = tuple(raw_data.dims) if has_dims else ()

    # Determine progress bar steps: per-channel if iter_axis present, else per-harmonic
    if iter_axis is not None and iter_axis in raw_dims:
        try:
            iter_axis_index = raw_dims.index(iter_axis)
            n_steps = int(raw_data.shape[iter_axis_index])
        except (ValueError, IndexError, TypeError):
            n_steps = 1
    else:
        n_steps = len(harmonics) if isinstance(harmonics, (list, tuple)) else 1

    pbr = show_activity_progress(
        desc=progress_description or f"Reading {filename}...",
        total=n_steps + 1,
    )

    try:
        if iter_axis is None or iter_axis not in raw_dims:
            # Handle files without iteration axis or when keepdims=False squeezed it out
            if axis_override is not None:
                axis = axis_override
            elif file_extension in [".tif", ".tiff"]:
                axis = 0
            elif has_dims and "H" in raw_dims:
                axis = raw_dims.index("H")
            elif has_dims and "C" in raw_dims:
                axis = raw_dims.index("C")
            else:
                axis = 0

            if file_extension in [".lsm", ".tif", ".tiff"]:
                axes_to_sum = tuple(range(1, len(raw_data.shape)))
            else:
                axes_to_sum = tuple(
                    i for i in range(len(raw_data.shape)) if i != axis
                )

            pbr.set_description("Summing signal...")
            pbr.update(1)
            summed_signal = np.sum(raw_data, axis=axes_to_sum)

            if hasattr(summed_signal, 'values'):
                summed_signal = summed_signal.values

            # Only set channel for files that actually have channels (FLIM files)
            if file_extension not in [".lsm", ".tif", ".tiff"]:
                settings['channel'] = 0

            # Determine number of histogram samples along selected axis
            try:
                n_samples = raw_data.shape[axis]
            except (IndexError, TypeError, AttributeError):
                try:
                    n_samples = raw_data.values.shape[axis]
                except (IndexError, TypeError, AttributeError):
                    n_samples = 0

            try:
                harmonics_to_use = _clamp_harmonics(harmonics, int(n_samples))
            except (ValueError, TypeError) as e:
                show_error(str(e))
                return []

            pbr.set_description("Computing phasor transform...")
            mean_intensity_image, G_image, S_image = phasor_from_signal(
                raw_data, axis=axis, harmonic=harmonics_to_use
            )
            pbr.update(n_steps)
            channel_suffix = " Intensity Image"
            add_kwargs = {
                "name": f"{filename}{channel_suffix}",
                "metadata": {
                    "original_mean": mean_intensity_image,
                    "settings": settings,
                    "summed_signal": (
                        summed_signal.tolist()
                        if hasattr(summed_signal, 'tolist')
                        else summed_signal
                    ),
                    "G": G_image,
                    "S": S_image,
                    "G_original": G_image.copy(),
                    "S_original": S_image.copy(),
                    "harmonics": harmonics_to_use,
                },
            }
            if keep_signal:
                add_kwargs["metadata"]["signal_full"] = np.asarray(raw_data)
                add_kwargs["metadata"]["signal_axis"] = int(axis)
            layers.append((mean_intensity_image, add_kwargs))
        else:
            # Handle multi-channel files with iteration axis
            iter_axis_index = raw_data.dims.index(iter_axis)
            channel_coord = raw_data.coords.get(iter_axis)
            if (
                channel_coord is not None
                and len(channel_coord) == raw_data.shape[iter_axis_index]
            ):
                channel_labels = list(channel_coord.values)
            else:
                channel_labels = list(range(raw_data.shape[iter_axis_index]))

            n_channels = len(channel_labels)
            for channel_pos, channel_label in enumerate(channel_labels):
                pbr.set_description(f"Channel {channel_pos + 1}/{n_channels}")
                pbr.update(1)
                channel_data = raw_data.isel({iter_axis: channel_pos})
                # Allow override of histogram axis via reader options
                if axis_override is not None:
                    histogram_axis = axis_override
                else:
                    histogram_axis = (
                        channel_data.dims.index("H")
                        if "H" in channel_data.dims
                        else 0
                    )

                # Calculate summed signal over spatial dimensions for this channel
                axes_to_sum = tuple(
                    i
                    for i in range(len(channel_data.shape))
                    if i != histogram_axis
                )
                summed_signal = np.sum(channel_data, axis=axes_to_sum)

                # Convert xarray DataArray to numpy array before converting to list
                if hasattr(summed_signal, 'values'):
                    summed_signal = summed_signal.values

                # Create settings dict for this channel
                channel_settings = settings.copy()
                try:
                    channel_settings['channel'] = int(
                        np.asarray(channel_label).item()
                    )
                except (TypeError, ValueError):
                    channel_settings['channel'] = channel_label

                # Determine samples for this histogram axis and clamp harmonics
                try:
                    n_samples = channel_data.shape[histogram_axis]
                except (IndexError, TypeError, AttributeError):
                    try:
                        n_samples = channel_data.values.shape[histogram_axis]
                    except (IndexError, TypeError, AttributeError):
                        n_samples = 0

                try:
                    harmonics_to_use = _clamp_harmonics(
                        harmonics, int(n_samples)
                    )
                except (ValueError, TypeError) as e:
                    show_error(str(e))
                    return []

                mean_intensity_image, G_image, S_image = phasor_from_signal(
                    channel_data,
                    axis=histogram_axis,
                    harmonic=harmonics_to_use,
                )
                add_kwargs = {
                    "name": f"{filename} Intensity Image: Channel {channel_label}",
                    "metadata": {
                        "original_mean": mean_intensity_image,
                        "settings": channel_settings,
                        "summed_signal": (
                            summed_signal.tolist()
                            if hasattr(summed_signal, 'tolist')
                            else summed_signal
                        ),
                        "G": G_image,
                        "S": S_image,
                        "G_original": G_image.copy(),
                        "S_original": S_image.copy(),
                        "harmonics": harmonics_to_use,
                    },
                }
                if keep_signal:
                    add_kwargs["metadata"]["signal_full"] = np.asarray(
                        channel_data
                    )
                    add_kwargs["metadata"]["signal_axis"] = int(histogram_axis)
                layers.append((mean_intensity_image, add_kwargs))
    finally:
        pbr.close()

    # Set colormaps if multichannel image
    if len(layers) == 2:
        # add colormaps MAGENTA_GREEN
        for layer, cmap in zip(layers, MAGENTA_GREEN, strict=False):
            layer[1]["colormap"] = cmap
            layer[1]['blending'] = 'additive'
    elif len(layers) > 2:
        # add colormaps CYMRGB in a cycle
        for layer, cmap in zip(layers, itertools.cycle(CYMRGB)):
            layer[1]["colormap"] = cmap
            layer[1]['blending'] = 'additive'

    return layers


def raw_file_stack_reader(
    paths: list[str],
    reader_options: dict | None = None,
    harmonics: Union[int, Sequence[int], None] = None,
) -> list[tuple]:
    """Read multiple raw data files and stack them into a 3D volume.

    Each file is treated as one spatial slice along the new first axis.
    All files must share the same extension and produce layers with
    identical spatial dimensions.

    Parameters
    ----------
    paths : list of str
        Ordered list of file paths (one per slice).
    reader_options : dict, optional
        Reader options forwarded to each single-file reader call.
    harmonics : Union[int, Sequence[int], None], optional
        Harmonics to compute.  Defaults to ``[1, 2]``.

    Returns
    -------
    layer_data : list of tuples
        Napari layer-data tuples with 3D arrays (stack, Y, X).

    Raises
    ------
    ValueError
        If files have mismatched extensions or spatial shapes.
    """
    if not paths:
        show_error("No files provided for stacking.")
        return []

    # Validate consistent extensions
    extensions = set()
    for p in paths:
        _, ext = _get_filename_extension(p)
        extensions.add(ext)
    if len(extensions) > 1:
        show_error(
            f"All files must share the same extension, got: {extensions}"
        )
        return []

    # Read the files concurrently. Each read decodes one file's signal, so
    # the pool is sized against free memory as well as core count.
    largest = 0
    for p in paths:
        with suppress(OSError):
            largest = max(largest, os.path.getsize(p))
    stack_workers = workers_for_memory(largest, n_items=len(paths))

    pbr = show_activity_progress(
        desc=f"Reading {len(paths)} file(s)...", total=len(paths)
    )
    try:
        per_file_layers: list[list[tuple]] = parallel_map(
            lambda p: raw_file_reader(
                p, reader_options=reader_options, harmonics=harmonics
            ),
            paths,
            workers=stack_workers,
            progress=lambda index: pbr.update(1),
        )
    finally:
        pbr.close()

    # Determine how many channels the first file produced
    n_channels = len(per_file_layers[0])

    # Verify every file produced the same number of channels
    for idx, file_layers in enumerate(per_file_layers):
        if len(file_layers) != n_channels:
            show_error(
                f"File {paths[idx]} produced {len(file_layers)} channel(s) "
                f"but the first file produced {n_channels}. "
                "All files must have the same number of channels."
            )
            return []

    # Stack per-channel across files
    stacked_layers = []
    for ch in range(n_channels):
        # Collect arrays for this channel across all files
        means = []
        g_arrays = []
        s_arrays = []
        g_orig_arrays = []
        s_orig_arrays = []
        summed_signals = []

        ref_shape = per_file_layers[0][ch][0].shape
        for file_idx, file_layers in enumerate(per_file_layers):
            data, kwargs = file_layers[ch]
            if data.shape != ref_shape:
                show_error(
                    f"Spatial shape mismatch: file {paths[file_idx]} has "
                    f"shape {data.shape} but expected {ref_shape}."
                )
                return []

            means.append(data)

            meta = kwargs["metadata"]
            g_arrays.append(meta["G"])
            s_arrays.append(meta["S"])
            g_orig_arrays.append(meta["G_original"])
            s_orig_arrays.append(meta["S_original"])

            sig = meta.get("summed_signal")
            if sig is not None:
                if isinstance(sig, list):
                    sig = np.array(sig)
                summed_signals.append(sig)

        # Stack along new axis 0 → (n_files, Y, X)
        stacked_mean = np.stack(means, axis=0)

        # G and S may have shape (n_harmonics, Y, X) or (Y, X)
        # We stack along a new axis: if 3D → (n_harmonics, n_files, Y, X)
        #                            if 2D → (n_files, Y, X)
        g_sample = g_arrays[0]
        if g_sample.ndim >= 3:
            # (n_harmonics, Y, X) → stack each harmonic's slices
            stacked_g = np.stack(g_arrays, axis=1)
            stacked_s = np.stack(s_arrays, axis=1)
            stacked_g_orig = np.stack(g_orig_arrays, axis=1)
            stacked_s_orig = np.stack(s_orig_arrays, axis=1)
        else:
            stacked_g = np.stack(g_arrays, axis=0)
            stacked_s = np.stack(s_arrays, axis=0)
            stacked_g_orig = np.stack(g_orig_arrays, axis=0)
            stacked_s_orig = np.stack(s_orig_arrays, axis=0)

        # Build metadata from the first file's channel metadata
        first_meta = per_file_layers[0][ch][1]["metadata"]
        first_kwargs = per_file_layers[0][ch][1]

        # Use a descriptive stack name
        common_dir = os.path.dirname(paths[0])
        dir_name = os.path.basename(common_dir) or "stack"
        channel_suffix = first_kwargs["name"].split("Intensity Image")[-1]
        stack_name = f"{dir_name} Stack Intensity Image{channel_suffix}"

        stack_meta = {
            "original_mean": stacked_mean.copy(),
            "settings": first_meta.get("settings", {}),
            "summed_signal": (
                [
                    s.tolist() if hasattr(s, 'tolist') else s
                    for s in summed_signals
                ]
                if summed_signals
                else None
            ),
            "G": stacked_g,
            "S": stacked_s,
            "G_original": stacked_g_orig,
            "S_original": stacked_s_orig,
            "harmonics": first_meta.get("harmonics"),
            "stack_files": [os.path.basename(p) for p in paths],
        }

        add_kwargs = {"name": stack_name, "metadata": stack_meta}

        # Preserve colormap / blending if set
        if "colormap" in first_kwargs:
            add_kwargs["colormap"] = first_kwargs["colormap"]
        if "blending" in first_kwargs:
            add_kwargs["blending"] = first_kwargs["blending"]

        stacked_layers.append((stacked_mean, add_kwargs))

    return stacked_layers


#: Dimensions that hold mosaic tiles, in the order they are preferred when
#: detecting a file's tile axis. ``M`` is the CZI mosaic axis, ``V`` a view,
#: ``B`` an acquisition block and ``S`` a scene.
TILE_AXIS_CANDIDATES = ("M", "V", "B", "S")

#: Dimensions that never hold tiles: the two spatial axes, plus ``C``/``H``/
#: ``Q``, which carry the channel or the histogram the phasor is computed
#: from.
NON_TILE_AXES = frozenset({"Y", "X", "C", "H", "Q"})


def probe_tile_axes(path, reader_options=None):
    """Report which dimensions of a file could hold mosaic tiles.

    A mosaic is often stored as a single file with its tiles along one
    dimension, such as the ``M`` axis of a Zeiss CZI. This inspects a file
    and reports the candidates, so the caller can offer a choice rather than
    guessing.

    For CZI the dimension sizes are taken from the file's sub-block
    directory, which is what the pixel reader builds its axes from, so no
    pixel data has to be read. Other formats fall back to reading the signal.

    Parameters
    ----------
    path : str
        Path to the file to inspect.
    reader_options : dict, optional
        Reader options, used only by the fallback path.

    Returns
    -------
    dict
        Maps axis to size, for every axis larger than one that could hold
        tiles. Keys are dimension names, or integer positions for formats
        such as TIFF whose axes are unnamed. Empty if the file holds a single
        tile or could not be inspected. Ordered with the recognized mosaic
        dimensions first.
    """
    # A CZI mosaic keeps its tiles in positioned sub-blocks rather than along
    # a dimension, so it has to be recognized before looking at the axes.
    mosaic = czi_mosaic_info(path)
    if mosaic is not None:
        return {CZI_MOSAIC_AXIS: mosaic["n_tiles"]}

    sizes = file_axis_sizes(path, reader_options)
    if not sizes:
        return {}

    # An unnamed format's last three axes are the histogram and the two
    # spatial axes, so only the ones before them can hold tiles.
    positional = [name for name in sizes if isinstance(name, int)]
    leading = set(positional[:-3]) if positional else set()

    candidates = {
        name: size
        for name, size in sizes.items()
        if size > 1
        and (
            name in leading
            if isinstance(name, int)
            else name not in NON_TILE_AXES
        )
    }
    ordered = {
        name: candidates.pop(name)
        for name in TILE_AXIS_CANDIDATES
        if name in candidates
    }
    ordered.update(candidates)
    return ordered


def file_axis_sizes(path, reader_options=None):
    """Return every axis of a file and its size, or an empty dict.

    Used both to find a file's tile axis and to explain, when none is found,
    what the file actually contains.
    """
    _, extension = _get_filename_extension(path)

    sizes = None
    if extension == ".czi":
        sizes = _czi_dimension_sizes(path)
    if sizes is None:
        sizes = _signal_dimension_sizes(path, reader_options)
    return sizes or {}


def describe_file_axes(path, reader_options=None):
    """Return a readable summary of a file's axes, for error messages."""
    sizes = file_axis_sizes(path, reader_options)
    if not sizes:
        return "unknown"
    return ", ".join(
        f"{describe_tile_axis(axis)}={size}" for axis, size in sizes.items()
    )


def describe_tile_axis(axis):
    """Return a human-readable name for a tile axis key."""
    labels = {
        CZI_MOSAIC_AXIS: "Mosaic tiles",
        "M": "M (mosaic)",
        "V": "V (view)",
        "B": "B (block)",
        "S": "S (scene)",
        "T": "T (time)",
        "Z": "Z (depth)",
    }
    if isinstance(axis, (int, np.integer)):
        return f"Axis {int(axis)}"
    return labels.get(axis, str(axis))


#: Key used to denote a CZI mosaic, whose tiles are stored as positioned
#: sub-blocks rather than along a named dimension.
CZI_MOSAIC_AXIS = "mosaic"


class CziMosaic:
    """Access the tiles of a Zeiss CZI mosaic one at a time.

    A CZI mosaic does not store its tiles along a dimension. Each tile is a
    group of sub-blocks carrying their own position in the mosaic, and the
    file's nominal ``Y``/``X`` extent spans the whole scanned area. Reading
    such a file whole is usually impossible: a slide scan of a few hundred
    tiles easily implies an array of many terabytes, nearly all of it empty.

    This reads the sub-block directory, which only touches the file's index,
    and then decodes one tile at a time.

    Parameters
    ----------
    path : str
        Path to the CZI file.

    Attributes
    ----------
    positions : list of tuple
        ``(y, x)`` pixel position of each tile, relative to the top-left of
        the mosaic.
    tile_shape : tuple of int
        ``(height, width)`` of one tile, in pixels.
    """

    def __init__(self, path):
        import czifile

        self.path = path
        self._czi = czifile.CziFile(path)
        entries = self._czi.filtered_subblock_directory
        if not entries:
            raise ValueError(f"{os.path.basename(path)} has no image data.")

        dims = entries[0].dims
        self._y = dims.index("Y")
        self._x = dims.index("X")
        self._channel = dims.index("C") if "C" in dims else None

        grouped: dict[int, list] = {}
        for entry in entries:
            grouped.setdefault(int(entry.mosaic_index), []).append(entry)
        self._tiles = [grouped[key] for key in sorted(grouped)]

        first = entries[0]
        self.tile_shape = (
            int(first.shape[self._y]),
            int(first.shape[self._x]),
        )

        raw = [
            (int(tile[0].start[self._y]), int(tile[0].start[self._x]))
            for tile in self._tiles
        ]
        min_y = min(position[0] for position in raw)
        min_x = min(position[1] for position in raw)
        self.positions = [(y - min_y, x - min_x) for y, x in raw]

    @property
    def n_tiles(self):
        """Number of tiles in the mosaic."""
        return len(self._tiles)

    @property
    def n_channels(self):
        """Number of channel planes each tile holds."""
        return len(self._tiles[0]) if self._tiles else 0

    def read_tile(self, index, binning=1):
        """Return one tile as a ``(C, Y, X)`` array.

        Parameters
        ----------
        index : int
            Tile position in the mosaic.
        binning : int, optional
            Bin the tile spatially by this factor. Binning sums the photons
            of each block, so the phasor coordinates of a binned tile are the
            photon-weighted average of the pixels that went into it, exactly
            as if the detector had had larger pixels.

        Returns
        -------
        xarray.DataArray
            Dimensions ``('C', 'Y', 'X')``.
        """
        if not 0 <= index < self.n_tiles:
            raise ValueError(
                f"{os.path.basename(self.path)} has {self.n_tiles} tile(s); "
                f"cannot read tile {index}."
            )

        entries = self._tiles[index]
        if self._channel is not None:
            entries = sorted(entries, key=lambda e: e.start[self._channel])

        planes = [
            np.asarray(entry.read_segment_data(self._czi).data()).reshape(
                self.tile_shape
            )
            for entry in entries
        ]
        cube = np.stack(planes)
        cube = _bin_spatial(cube, binning)
        return xr.DataArray(cube, dims=("C", "Y", "X"))

    def binned_positions(self, binning=1):
        """Return the tile positions in the binned pixel grid."""
        binning = max(1, int(binning))
        if binning == 1:
            return list(self.positions)
        return [(y // binning, x // binning) for y, x in self.positions]

    def binned_tile_shape(self, binning=1):
        """Return the tile size after binning."""
        binning = max(1, int(binning))
        return (
            self.tile_shape[0] // binning,
            self.tile_shape[1] // binning,
        )

    def canvas_shape(self, binning=1):
        """Return the stitched canvas size for a binning factor."""
        height, width = self.binned_tile_shape(binning)
        positions = self.binned_positions(binning)
        return (
            max(y for y, _ in positions) + height,
            max(x for _, x in positions) + width,
        )

    def close(self):
        """Close the underlying file."""
        with suppress(Exception):
            self._czi.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()


def _bin_spatial(cube, factor):
    """Sum ``(C, Y, X)`` *cube* over ``factor`` x ``factor`` spatial blocks."""
    factor = max(1, int(factor))
    if factor == 1:
        return cube

    channels, height, width = cube.shape
    height -= height % factor
    width -= width % factor
    if height <= 0 or width <= 0:
        raise ValueError(
            f"Binning by {factor} leaves nothing of a "
            f"{cube.shape[1]}x{cube.shape[2]} tile."
        )
    trimmed = cube[:, :height, :width]
    # Photon counts are summed rather than averaged, so the phasor of the
    # binned tile stays the photon-weighted phasor of its pixels.
    return trimmed.reshape(
        channels, height // factor, factor, width // factor, factor
    ).sum(axis=(2, 4), dtype=np.uint32)


def czi_mosaic_info(path):
    """Describe a CZI mosaic without reading any pixels, or return ``None``.

    Returns
    -------
    dict or None
        ``{'n_tiles', 'tile_shape', 'canvas_shape', 'n_channels'}``, or
        ``None`` when the file is not a CZI mosaic of several tiles.
    """
    _, extension = _get_filename_extension(path)
    if extension != ".czi":
        return None
    try:
        with CziMosaic(path) as mosaic:
            if mosaic.n_tiles < 2:
                return None
            return {
                "n_tiles": mosaic.n_tiles,
                "tile_shape": mosaic.tile_shape,
                "canvas_shape": mosaic.canvas_shape(),
                "n_channels": mosaic.n_channels,
            }
    except Exception:  # noqa: BLE001 - probing is best effort
        return None


def _czi_dimension_sizes(path):
    """Return CZI dimension sizes from the sub-block directory, or ``None``.

    Reading the directory only touches the file's index, not its pixels.
    """
    try:
        import czifile

        sizes: dict[str, int] = {}
        with czifile.CziFile(path) as czi:
            for block in czi.filtered_subblock_directory:
                for name, start, size in zip(
                    block.dims, block.start, block.shape, strict=False
                ):
                    sizes[name] = max(
                        sizes.get(name, 0), int(start) + int(size)
                    )
        # phasorpy renames the CZI phase axis to avoid clashing with the
        # lifetime histogram axis; mirror that so names match the signal.
        if "H" in sizes:
            sizes["Q"] = sizes.pop("H")
        return sizes
    except Exception:  # noqa: BLE001 - probing is best effort
        return None


def _signal_dimension_sizes(path, reader_options=None):
    """Return every axis of a raw signal and its size, or ``None``.

    Named dimensions are keyed by name; formats that return a plain array are
    keyed by integer position instead.
    """
    try:
        _, _, io_options = _split_widget_reader_options(reader_options)
        signal = load_raw_signal(path, io_options)
        shape = np.shape(signal)
        if hasattr(signal, "dims"):
            return {
                str(name): int(size)
                for name, size in zip(signal.dims, shape, strict=True)
            }
        return {index: int(size) for index, size in enumerate(shape)}
    except Exception:  # noqa: BLE001 - probing is best effort
        return None


def _read_file_tiles(
    path,
    indices,
    reader_options=None,
    harmonics=None,
    tile_axis=None,
    progress=None,
):
    """Read the requested tiles out of one file.

    Files contributing a single tile go through the normal reader, so mosaics
    of already-transformed files keep working. Files contributing several
    tiles are read once and sliced along their tile axis.

    Parameters
    ----------
    path : str
        File to read.
    indices : sequence of int
        Tile positions wanted from this file, possibly with repeats.
    reader_options : dict, optional
        Reader options.
    harmonics : int or sequence of int, optional
        Harmonics to compute.
    tile_axis : str, optional
        Dimension holding the tiles. Detected when ``None``.
    progress : optional
        Progress bar updated once per tile.

    Returns
    -------
    dict
        Maps tile index to that tile's list of per-channel layer tuples.

    Raises
    ------
    ValueError
        If no reader is available, if a multi-tile read finds no usable tile
        axis, or if an index is out of range.
    """
    name = os.path.basename(path)
    wanted = sorted({int(index) for index in indices})

    # Read the file whole only when nothing asks for it to be split: no tile
    # axis was chosen and only tile 0 is wanted. This keeps mosaics of
    # already-transformed files working, while an explicit tile axis always
    # slices, even when just one tile is read from the file.
    if tile_axis is None and wanted == [0]:
        if progress is not None:
            progress.set_description(f"Reading {name}")
        # Dispatch through napari_get_reader so mosaics of already-transformed
        # files (OME-TIFF, SimFCS, ISS) stitch as well as raw acquisitions.
        reader = napari_get_reader(
            path, reader_options=reader_options, harmonics=harmonics
        )
        if reader is None:
            raise ValueError(f"No reader available for {path}.")
        result = {0: reader(path)}
        if progress is not None:
            progress.update(1)
        return result

    binning = int((reader_options or {}).get("binning", 1) or 1)
    axis_override, keep_signal, io_options = _split_widget_reader_options(
        reader_options
    )
    _, extension = _get_filename_extension(path)

    if tile_axis == CZI_MOSAIC_AXIS or (
        tile_axis is None and czi_mosaic_info(path) is not None
    ):
        return _read_czi_mosaic_tiles(
            path,
            wanted,
            harmonics=harmonics,
            binning=binning,
            keep_signal=keep_signal,
            progress=progress,
        )

    if extension not in extension_mapping["raw"]:
        raise ValueError(
            f"{name} holds processed data, which cannot be split into tiles. "
            "Select one file per tile instead."
        )

    if progress is not None:
        progress.set_description(f"Reading {name}")
    signal = load_raw_signal(path, io_options)

    axis_position = _resolve_tile_axis(signal, tile_axis, name)
    n_available = int(np.shape(signal)[axis_position])
    out_of_range = [index for index in wanted if not 0 <= index < n_available]
    if out_of_range:
        raise ValueError(
            f"{name} has {n_available} tile(s) along the tile axis; "
            f"cannot read tile(s) {out_of_range}."
        )

    # Slicing removes the tile axis, so a phasor axis given by position and
    # sitting after it shifts down by one.
    if axis_override is not None and axis_override > axis_position:
        axis_override -= 1

    filename = _get_filename_extension(path)[0]
    layers_by_index = {}
    try:
        # The signal is already in memory, so slicing a tile out of it and
        # transforming it is independent per tile and parallelizes cleanly.
        # Each tile only allocates its own (small) phasor output.
        def transform(item):
            position, index = item
            return _phasor_layers_from_signal(
                _take_tile(signal, axis_position, index),
                filename=f"{filename} [{index}]",
                file_extension=extension,
                harmonics=harmonics if harmonics is not None else [1, 2],
                axis_override=axis_override,
                keep_signal=keep_signal,
                progress_description=f"{name}: tile {position + 1}",
            )

        def report(position):
            if progress is not None:
                progress.set_description(
                    f"{name}: tile {position + 1}/{len(wanted)}"
                )
                progress.update(1)

        results = parallel_map(
            transform, list(enumerate(wanted)), progress=report
        )
        layers_by_index = dict(zip(wanted, results, strict=True))
    finally:
        # Release the file's signal as soon as its tiles are transformed while
        # keeping the name in scope for the nested tile-transform closure.
        signal = None

    return layers_by_index


def _read_czi_mosaic_tiles(
    path,
    wanted,
    harmonics=None,
    binning=1,
    keep_signal=False,
    progress=None,
):
    """Phasor-transform the requested tiles of a CZI mosaic.

    Tiles are decoded one at a time straight from their sub-blocks, so only
    one tile's spectral stack is ever held in memory.
    """
    name = os.path.basename(path)
    filename = _get_filename_extension(path)[0]
    layers_by_index = {}

    with CziMosaic(path) as mosaic:
        out_of_range = [
            index for index in wanted if not 0 <= index < mosaic.n_tiles
        ]
        if out_of_range:
            raise ValueError(
                f"{name} has {mosaic.n_tiles} tile(s); cannot read tile(s) "
                f"{out_of_range}."
            )

        # Decoding pulls sub-blocks through one shared file handle, which is
        # not thread-safe, so decodes are serialized behind a lock while the
        # phasor transforms -- the expensive half -- overlap freely.
        decode_lock = threading.Lock()

        def transform(item):
            position, index = item
            with decode_lock:
                tile = mosaic.read_tile(index, binning=binning)
            return _phasor_layers_from_signal(
                tile,
                filename=f"{filename} [{index}]",
                file_extension=".czi",
                harmonics=harmonics if harmonics is not None else [1, 2],
                keep_signal=keep_signal,
                progress_description=f"{name}: tile {position + 1}",
            )

        def report(position):
            if progress is not None:
                progress.set_description(
                    f"{name}: tile {position + 1}/{len(wanted)}"
                )
                progress.update(1)

        results = parallel_map(
            transform, list(enumerate(wanted)), progress=report
        )
        layers_by_index = dict(zip(wanted, results, strict=True))

    return layers_by_index


def _resolve_tile_axis(signal, tile_axis, name):
    """Return the positional index of a signal's tile axis.

    *tile_axis* may be a dimension name, a positional index, or ``None`` to
    detect one. Formats such as TIFF return plain arrays with no dimension
    names, so a positional index is the only way to address their axes.

    Raises
    ------
    ValueError
        If the axis cannot be found or does not exist in the signal.
    """
    dims = tuple(str(dim) for dim in getattr(signal, "dims", ()))
    shape = np.shape(signal)

    if isinstance(tile_axis, (int, np.integer)) and not isinstance(
        tile_axis, bool
    ):
        axis = int(tile_axis)
        if not -len(shape) <= axis < len(shape):
            raise ValueError(
                f"{name} has {len(shape)} dimension(s); axis {tile_axis} is "
                "out of range."
            )
        return axis % len(shape)

    if tile_axis is not None:
        if tile_axis not in dims:
            raise ValueError(
                f"{name} has no dimension named {tile_axis!r}. Available "
                f"dimensions: {', '.join(dims) or 'none (unnamed axes)'}."
            )
        return dims.index(tile_axis)

    candidates = {
        dim: index
        for index, dim in enumerate(dims)
        if shape[index] > 1 and dim not in NON_TILE_AXES
    }
    for candidate in TILE_AXIS_CANDIDATES + tuple(candidates):
        if candidate in candidates:
            return candidates[candidate]

    raise ValueError(
        f"{name} has no dimension holding tiles. Available dimensions: "
        f"{', '.join(dims) or 'none (unnamed axes)'}. Choose the tile axis "
        "explicitly, or select one file per tile."
    )


def _take_tile(signal, axis_position, index):
    """Return one tile of *signal*, dropping the tile axis."""
    if hasattr(signal, "isel"):
        return signal.isel({str(signal.dims[axis_position]): index})
    return np.take(signal, index, axis=axis_position)


class TileSet:
    """Per-tile phasor coordinates of a mosaic, cached for re-stitching.

    Reading and phasor-transforming the tiles is the expensive part of
    importing a mosaic; placing and blending them is comparatively cheap.
    Holding the transformed tiles in a :class:`TileSet` lets the overlap be
    re-tuned interactively, since :meth:`stitch` can be called any number of
    times without touching the files again.

    Memory is ``n_tiles * (1 + 2 * n_harmonics)`` single precision planes,
    which for FLIM data is typically one to two orders of magnitude less than
    the raw signal the tiles were computed from.

    Parameters
    ----------
    sources : list of TileSource
        Where each tile came from, in placement order.
    tile_shape : tuple of int
        ``(height, width)`` shared by every tile.
    tiles : list of list of tuple
        ``tiles[channel][index]`` is the ``(mean, G, S)`` triple of one tile.
    templates : list of dict
        Per-channel ``add_kwargs`` taken from the first tile, used as the
        basis for the stitched layer's metadata.
    """

    def __init__(
        self, sources, tile_shape, tiles, templates, summed_signals=None
    ):
        self.sources = as_tile_sources(sources)
        self.tile_shape = tuple(tile_shape)
        self.tiles = tiles
        self.templates = templates
        self.summed_signals = summed_signals or [None] * len(tiles)

    @property
    def paths(self):
        """Paths the tiles came from, in placement order, with repeats."""
        return [source.path for source in self.sources]

    @property
    def n_channels(self):
        """Number of channels each tile produced."""
        return len(self.tiles)

    @property
    def n_tiles(self):
        """Number of tiles in the mosaic."""
        return len(self.sources)

    @property
    def n_files(self):
        """Number of distinct files the tiles came from."""
        return len(set(self.paths))

    def means(self, channel=0):
        """Return the mean intensity image of every tile for *channel*."""
        return [tile[0] for tile in self.tiles[channel]]

    def nbytes(self):
        """Return the total size of the cached arrays, in bytes."""
        return sum(
            array.nbytes
            for channel in self.tiles
            for tile in channel
            for array in tile
        )

    def stitch(self, geometry, progress=None):
        """Blend the cached tiles into napari layer-data tuples.

        Parameters
        ----------
        geometry : TileGeometry
            Mosaic layout. Its ``tile_shape`` is taken from this tile set, so
            only the placements, overlap and blend mode need to be set.
        progress : callable, optional
            Called with ``(channel, tile_index)`` as blending proceeds.

        Returns
        -------
        list of tuple
            One ``(data, add_kwargs)`` tuple per channel.
        """
        geometry = replace(geometry, tile_shape=self.tile_shape)

        layers = []
        for channel in range(self.n_channels):
            mean, real, imag, coverage = blend_phasor_tiles(
                self.tiles[channel],
                geometry,
                progress=(
                    None
                    if progress is None
                    else lambda index, ch=channel: progress(ch, index)
                ),
            )

            template = self.templates[channel]
            template_meta = template.get("metadata", {})

            summed_signal = self.summed_signals[channel]
            metadata = {
                "original_mean": mean.copy(),
                "settings": template_meta.get("settings", {}),
                "summed_signal": (
                    summed_signal.tolist()
                    if summed_signal is not None
                    else template_meta.get("summed_signal")
                ),
                "G": real,
                "S": imag,
                "G_original": real.copy(),
                "S_original": imag.copy(),
                "harmonics": template_meta.get("harmonics"),
                "tile_files": [source.label for source in self.sources],
                "tile_geometry": geometry.to_dict(),
                "tile_coverage": coverage,
            }

            # A mosaic spread over many files is named after their folder; one
            # held inside a single file is named after that file.
            if self.n_files == 1:
                stem = _get_filename_extension(self.paths[0])[0]
            else:
                stem = os.path.basename(os.path.dirname(self.paths[0]))
            channel_suffix = template["name"].split("Intensity Image")[-1]
            name = f"{stem or 'mosaic'} Mosaic Intensity Image{channel_suffix}"

            add_kwargs = {"name": name, "metadata": metadata}
            for key in ("colormap", "blending"):
                if key in template:
                    add_kwargs[key] = template[key]
            layers.append((mean, add_kwargs))

        return layers


def read_tile_phasors(
    tiles: list,
    reader_options: dict | None = None,
    harmonics: Union[int, Sequence[int], None] = None,
    tile_axis: str | None = None,
) -> "TileSet":
    """Read every tile of a mosaic and phasor-transform it.

    Handles both ways a mosaic is stored: one file per tile, and a single
    file holding all its tiles along a dedicated dimension (see
    :func:`probe_tile_axes`). Mixtures of the two work as well.

    Each file is read exactly once and its raw signal released as soon as
    every tile in it has been transformed, so peak memory stays at one file's
    signal rather than the whole mosaic.

    Parameters
    ----------
    tiles : list
        Tiles in placement order, as paths, ``(path, index)`` pairs, or
        :class:`~napari_phasors._stitching.TileSource` objects.
    reader_options : dict, optional
        Reader options forwarded to each file reader call.
    harmonics : int or sequence of int, optional
        Harmonics to compute.
    tile_axis : str, optional
        Name of the dimension holding the tiles inside a multi-tile file, for
        example ``'M'`` for a CZI mosaic. Detected automatically when
        ``None``. Ignored for files contributing a single tile.

    Returns
    -------
    TileSet
        Cached phasor coordinates, ready to be stitched.

    Raises
    ------
    ValueError
        If no tiles are given, if the files do not share one extension, if a
        tile fails to produce layers, or if the tiles disagree on shape or
        channel count.
    """
    sources = as_tile_sources(tiles)
    if not sources:
        raise ValueError("No files provided for stitching.")

    extensions = {_get_filename_extension(s.path)[1] for s in sources}
    if len(extensions) > 1:
        raise ValueError(
            f"All tiles must share the same extension, got: {extensions}"
        )

    # Group by file, keeping first-appearance order, so each file is opened
    # once no matter how its tiles are ordered in the layout.
    per_file: dict[str, list[int]] = {}
    for source in sources:
        per_file.setdefault(source.path, []).append(source.index)

    tile_arrays: list[list[tuple]] = []
    templates: list[dict] = []
    summed_signals: list = []
    tile_shape = None
    n_channels = None
    frequencies = set()
    by_source: dict = {}

    items = list(per_file.items())

    # Files are read and transformed concurrently. The pool is sized against
    # free memory as well as core count, because N workers hold N files'
    # decoded signals at once where a sequential read held exactly one; the
    # on-disk size is a usable lower bound for that footprint.
    largest = 0
    for path, _ in items:
        with suppress(OSError):
            largest = max(largest, os.path.getsize(path))
    workers = workers_for_memory(largest, n_items=len(items))

    pbr = show_activity_progress(
        desc=f"Reading {len(sources)} tile(s)...", total=len(sources)
    )
    try:

        def read_file(item):
            path, indices = item
            # The napari progress bar is a Qt object belonging to this
            # thread, so workers never touch it; progress is reported below
            # as each file's results are collected.
            return _read_file_tiles(
                path,
                indices,
                reader_options=reader_options,
                harmonics=harmonics,
                tile_axis=tile_axis,
                progress=None,
            )

        def report(position):
            path, indices = items[position]
            pbr.set_description(f"Read {os.path.basename(path)}")
            pbr.update(len(indices))

        results = parallel_map(
            read_file, items, workers=workers, progress=report
        )

        for (path, _indices), layers_by_index in zip(
            items, results, strict=True
        ):
            name = os.path.basename(path)

            for index, layers in layers_by_index.items():
                if not layers:
                    raise ValueError(f"No data could be read from {path}.")

                if n_channels is None:
                    n_channels = len(layers)
                    tile_arrays = [[] for _ in range(n_channels)]
                    templates = [dict(layer[1]) for layer in layers]
                    summed_signals = [None] * n_channels
                elif len(layers) != n_channels:
                    raise ValueError(
                        f"{name} produced {len(layers)} channel(s) but the "
                        f"first tile produced {n_channels}. All tiles must "
                        "have the same number of channels."
                    )

                per_channel = []
                for channel, (mean, add_kwargs) in enumerate(layers):
                    mean = np.asarray(mean, dtype=np.float32)
                    if mean.ndim != 2:
                        raise ValueError(
                            f"Tile {name} is {mean.ndim}D; stitching expects "
                            "2D tiles. Pick the axis holding the tiles, or "
                            "select a single channel or slice."
                        )
                    if tile_shape is None:
                        tile_shape = mean.shape
                    elif mean.shape != tile_shape:
                        raise ValueError(
                            f"Shape mismatch: {name} has shape {mean.shape} "
                            f"but expected {tile_shape}."
                        )

                    metadata = add_kwargs["metadata"]
                    real = np.asarray(metadata["G"], dtype=np.float32)
                    imag = np.asarray(metadata["S"], dtype=np.float32)
                    if real.ndim == 2:
                        real = real[np.newaxis]
                        imag = imag[np.newaxis]
                    per_channel.append((mean, real, imag))

                    frequency = metadata.get("settings", {}).get("frequency")
                    if frequency is not None:
                        frequencies.add(round(float(frequency), 6))

                    # The mosaic's signal profile is the sum of its tiles',
                    # which keeps the signal preview and the harmonic limits
                    # meaningful.
                    signal = metadata.get("summed_signal")
                    if signal is not None:
                        signal = np.asarray(signal, dtype=np.float64)
                        accumulated = summed_signals[channel]
                        if accumulated is None:
                            summed_signals[channel] = signal
                        elif accumulated.shape == signal.shape:
                            summed_signals[channel] = accumulated + signal

                by_source[(path, index)] = per_channel
    finally:
        pbr.close()

    # Emit in placement order, which may differ from the order the files were
    # read in, and may repeat a tile.
    for source in sources:
        for channel, arrays in enumerate(
            by_source[(source.path, source.index)]
        ):
            tile_arrays[channel].append(arrays)

    if len(frequencies) > 1:
        show_error(
            "Tiles were acquired at different laser frequencies "
            f"({sorted(frequencies)}); the stitched phasor is not "
            "meaningful. Import them separately."
        )

    return TileSet(sources, tile_shape, tile_arrays, templates, summed_signals)


def raw_file_tile_reader(
    tiles: list,
    geometry,
    reader_options: dict | None = None,
    harmonics: Union[int, Sequence[int], None] = None,
    tile_axis: str | None = None,
) -> list[tuple]:
    """Read a set of tiles and stitch them into a single phasor image.

    Stitching happens in phasor space: the mean intensity and the G and S
    coordinates of each tile are blended with photon weighting, which gives
    exactly the result that summing the raw signals before the phasor
    transform would have produced, at a fraction of the memory.

    Parameters
    ----------
    tiles : list
        Tiles in the order matching ``geometry.placements``, as paths,
        ``(path, index)`` pairs, or
        :class:`~napari_phasors._stitching.TileSource` objects.
    geometry : TileGeometry
        Mosaic layout. ``tile_shape`` is filled in from the data.
    reader_options : dict, optional
        Reader options forwarded to each file reader call.
    harmonics : int or sequence of int, optional
        Harmonics to compute.
    tile_axis : str, optional
        Dimension holding the tiles inside a multi-tile file. Detected
        automatically when ``None``.

    Returns
    -------
    layer_data : list of tuple
        Napari layer-data tuples, one per channel. Returns an empty list and
        shows an error notification if the tiles could not be read.
    """
    try:
        tile_set = read_tile_phasors(
            tiles,
            reader_options=reader_options,
            harmonics=harmonics,
            tile_axis=tile_axis,
        )
    except ValueError as error:
        show_error(str(error))
        return []

    try:
        return tile_set.stitch(geometry)
    except ValueError as error:
        show_error(str(error))
        return []


def _infer_harmonics(
    requested: Union[int, Sequence[int], str, None],
    real: np.ndarray,
    mean: np.ndarray,
) -> Union[int, list[int]]:
    """Infer the harmonic numbers of already-computed phasor coordinates.

    Not every ``phasorpy.io`` reader for processed files reports which
    harmonics it returned: ``phasor_from_simfcs_referenced`` (R64/REF) and
    ``phasor_from_lif`` return no ``'harmonic'`` metadata. Leaving
    ``harmonics`` as None in the layer metadata makes downstream analyses
    treat the leading harmonic axis of G/S as image data -- e.g. component
    analysis then returns one fraction image per harmonic instead of one for
    the selected harmonic.

    Parameters
    ----------
    requested : int, sequence of int, 'all', or None
        The ``harmonic`` argument that was passed to the IO function.
    real : np.ndarray
        Real component of the phasor coordinates as returned by the reader.
    mean : np.ndarray
        Mean intensity image as returned by the reader.

    Returns
    -------
    int or list of int
        A single harmonic number when ``real`` has no harmonic axis,
        otherwise one number per plane of that axis.
    """
    if real.ndim == mean.ndim:
        # No harmonic axis: the reader was asked for a single harmonic.
        if isinstance(requested, (int, np.integer)) and not isinstance(
            requested, bool
        ):
            return int(requested)
        return 1

    n_harmonics = real.shape[0]
    if isinstance(requested, Sequence) and not isinstance(requested, str):
        with suppress(TypeError, ValueError):
            harmonics = [int(h) for h in requested]
            if len(harmonics) == n_harmonics:
                return harmonics
    # 'all', None, or a request that does not match what was read: the
    # readers return the file's harmonics in order, starting at the first.
    return list(range(1, n_harmonics + 1))


def _keep_first_harmonics(
    real: np.ndarray,
    imag: np.ndarray,
    harmonics: Union[int, Sequence[int]],
    mean_ndim: int,
    limit: int = 2,
) -> tuple[np.ndarray, np.ndarray, Union[int, list[int]]]:
    """Trim phasor coordinates to the first ``limit`` harmonics they hold.

    Used only when the caller requested no particular harmonic, so that
    processed files default to the same first-two-harmonics behaviour as raw
    files without a second read of the file.

    Parameters
    ----------
    real, imag : np.ndarray
        Phasor coordinates as returned by the reader. A leading axis of
        length > 1 is the harmonic axis.
    harmonics : int or sequence of int
        Harmonic number(s) the arrays hold, in the same order.
    mean_ndim : int
        Number of dimensions of the mean intensity image, used to tell a
        harmonic axis apart from an image axis.
    limit : int, optional
        Maximum number of harmonics to keep. Default is 2.

    Returns
    -------
    tuple
        ``(real, imag, harmonics)``, trimmed if there was anything to trim.
    """
    if real.ndim != mean_ndim + 1 or real.shape[0] <= limit:
        return real, imag, harmonics
    if not isinstance(harmonics, Sequence) or isinstance(harmonics, str):
        harmonics = np.atleast_1d(harmonics).tolist()
    return real[:limit], imag[:limit], list(harmonics)[:limit]


def processed_file_reader(
    path: str,
    reader_options: dict[str, str] | None = None,
    harmonics: Union[int, Sequence[int], None] = None,
) -> list[tuple]:
    """Reader function for files that contain processed images, as phasor
    coordinates or intensity images.

    Parameters
    ----------
    path : str
        Path to file.
    reader_options : dict, optional
        Dictionary containing the arguments to pass to the function.
    harmonics : Union[int, Sequence[int], None], optional
        Harmonic(s) to be processed. Can be a single integer, a sequence of
        integers, or None. Default is None, which reads the first two
        harmonics present in the file, or the first one if that is all it
        holds. Pass ``'all'`` to read every harmonic in the file.

    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples where each tuple in the list contains a
        napari.layers.Labels layer a tuple  (data, kwargs), where data is
        the mean intensity image as an array, and kwargs is a a dict of
        keyword arguments for the corresponding viewer.add_* method in napari,
        which contains the 'name' of the layer as well as the 'metadata',
        which is also a dict. The values for key 'phasor_features_labels_layer'
        in 'metadata' contain phasor coordinates as columns 'G' and 'S'.

    """
    # No explicit request: read everything the file holds, then keep only the
    # first two harmonics below. This matches the raw reader, whose default is
    # also the first two harmonics (see ``_clamp_harmonics``), and keeps files
    # that store many harmonics (IFLI, RE<n>, FLIM LABS JSON) from loading a
    # stack of them by default.
    default_harmonics = harmonics is None
    if harmonics is None:
        harmonics = 'all'
    filename, file_extension = _get_filename_extension(path)

    # Prepare reader options: remove widget-only keys and ensure harmonic present
    filtered_reader_options = reader_options.copy() if reader_options else {}
    filtered_reader_options.pop('phasor_axis', None)
    # Widget-level flag understood only by the raw reader; drop it so it is
    # never forwarded to the processed IO functions (which would reject it).
    filtered_reader_options.pop('_keep_signal', None)
    if 'harmonic' not in filtered_reader_options:
        filtered_reader_options['harmonic'] = harmonics

    pbr = show_activity_progress(desc=f"Loading {filename}...", total=3)
    try:
        mean_intensity_image, real, imag, attrs = extension_mapping[
            "processed"
        ][file_extension](path, filtered_reader_options)
        pbr.update(1)
        if "description" in attrs:
            # HTML-unescape the description to handle tifffile HTML encoding
            description_str = html.unescape(attrs["description"])
            description = json.loads(description_str)
            if len(json.dumps(description)) > 512 * 512:  # Threshold: 256 KB
                raise ValueError("Description dictionary is too large.")
            if "napari_phasors_settings" in description:
                settings = json.loads(description["napari_phasors_settings"])
                if "calibrated" in settings:
                    settings["calibrated"] = bool(settings["calibrated"])
        else:
            settings = {}
        if "frequency" in attrs:
            settings["frequency"] = attrs["frequency"]
        harmonics_read = attrs.get("harmonic", None)
        if harmonics_read is None:
            harmonics_read = _infer_harmonics(
                filtered_reader_options.get("harmonic"),
                real,
                mean_intensity_image,
            )

        if default_harmonics:
            real, imag, harmonics_read = _keep_first_harmonics(
                real, imag, harmonics_read, mean_intensity_image.ndim
            )

        original_mean_intensity_image = mean_intensity_image.copy()
        g_original = real.copy()
        s_original = imag.copy()

        should_apply_processing = False
        filter_params = {}
        threshold_value = 0
        threshold_upper_value = None

        if "filter" in settings:
            filter_settings = settings["filter"]
            if filter_settings.get("repeat", 0) > 0:
                should_apply_processing = True
                filter_params = {
                    "filter_method": filter_settings.get("method", "median"),
                    "size": filter_settings.get("size", 3),
                    "repeat": filter_settings.get("repeat", 1),
                    "sigma": filter_settings.get("sigma", 1.0),
                    "levels": filter_settings.get("levels", 3),
                }

        if "threshold" in settings and settings["threshold"] is not None:
            should_apply_processing = True
            threshold_value = settings["threshold"]

        if (
            "threshold_upper" in settings
            and settings["threshold_upper"] is not None
        ):
            should_apply_processing = True
            threshold_upper_value = settings["threshold_upper"]

        if should_apply_processing:
            pbr.set_description("Applying filters...")
            pbr.update(1)
            from ._utils import _apply_filter_and_threshold_to_phasor_arrays

            mean_intensity_image, real, imag = (
                _apply_filter_and_threshold_to_phasor_arrays(
                    mean_intensity_image,
                    real,
                    imag,
                    harmonics_read,
                    threshold=threshold_value,
                    threshold_upper=threshold_upper_value,
                    **filter_params,
                )
            )

            if "settings" not in settings:
                settings["settings"] = {}
            settings["filter"] = {
                "method": filter_params.get("filter_method", "median"),
                "size": filter_params.get("size", 3),
                "repeat": filter_params.get("repeat", 1),
                "sigma": filter_params.get("sigma", 1.0),
                "levels": filter_params.get("levels", 3),
            }
            settings["threshold"] = threshold_value
            if threshold_upper_value is not None:
                settings["threshold_upper"] = threshold_upper_value

        layers = []

        add_kwargs = {
            "name": filename + " Intensity Image",
            "metadata": {
                "original_mean": original_mean_intensity_image,
                "settings": settings,
                "G": real,
                "S": imag,
                "G_original": g_original,
                "S_original": s_original,
                "harmonics": harmonics_read,
            },
        }

        if "dims" in attrs:
            add_kwargs["axis_labels"] = tuple(attrs["dims"])
        elif "axes" in attrs:
            add_kwargs["axis_labels"] = tuple(attrs["axes"])

        z_spacing_um = settings.get("z_spacing_um")
        if z_spacing_um is not None and mean_intensity_image.ndim >= 3:
            try:
                z_idx = 0
                if "axis_labels" in add_kwargs:
                    labels = [
                        str(label).upper()
                        for label in add_kwargs["axis_labels"]
                    ]
                    if 'Z' in labels:
                        z_idx = labels.index('Z')
                scale = [1.0] * mean_intensity_image.ndim
                scale[z_idx] = float(z_spacing_um)
                add_kwargs["scale"] = tuple(scale)
            except (ValueError, TypeError):
                pass

        layers.append((mean_intensity_image, add_kwargs))
    finally:
        pbr.close()
    return layers


def _parse_and_call_io_function(
    path: str,
    func: Callable,
    args_defaults: dict[str, Any],
    reader_options: dict[str, Any] | None = None,
) -> Any:
    """Private helper function to parse arguments and call a `io` function.

    Parameters
    ----------
    path : str
        Path to file.
    func : callable
        Function to call.
    args_defaults : dict
        Dictionary containing the default arguments for the function.
    reader_options : dict, optional
        Dictionary containing the arguments to pass to the function.
        Default is None.

    Returns
    -------
    data : xarray.DataArray
        Data read from the file

    """
    args = {}
    # Use reader_options if provided, otherwise use the default
    if reader_options is not None:
        for arg, value in reader_options.items():
            args[arg] = value

    # Fill in defaults for any missing arguments not provided in reader_options
    for arg, (default, is_required) in args_defaults.items():
        if arg not in args:
            if is_required:
                raise ValueError(f"Required argument '{arg}' is missing.")
            args[arg] = default

    # Validate arguments against the function's signature
    valid_args = {}
    sig = inspect.signature(func)
    has_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in sig.parameters.values()
    )
    for arg, value in args.items():
        if arg in sig.parameters or has_kwargs:
            valid_args[arg] = value
        else:
            raise ValueError(
                f"Invalid argument '{arg}' for function {func.__name__}."
            )
    return func(path, **valid_args)


def _get_filename_extension(path: str) -> tuple[str, str]:
    """Get the filename and extension from a path.

    Parameters
    ----------
    path : str
        Path to file.

    Returns
    -------
    filename : str
        Filename.
    file_extension : str
        File extension including the leading dot.

    """
    filename = os.path.basename(path)
    parts = filename.split(".", 1)
    file_extension = "." + parts[1] if len(parts) > 1 else ""
    return parts[0], file_extension.lower()
