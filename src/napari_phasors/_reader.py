"""
This module contains functions to read files supported by `phasorpy.io`
and computes phasor coordinates with `phasorpy.phasor.phasor_from_signal`

"""

import html
import inspect
import itertools
import json
import os
from collections.abc import Callable, Sequence
from typing import Any, Union

import czifile
import numpy as np
import phasorpy.io as io
import tifffile
import xarray as xr
from napari.utils.colormaps.colormap_utils import CYMRGB, MAGENTA_GREEN
from napari.utils.notifications import show_error, show_info
from phasorpy.phasor import phasor_from_signal

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
        ".czi": lambda path, reader_options: read_czi(path),
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
    if path.endswith(tuple(extensions_both)):
        return lambda path: ambiguous_file_reader(
            path, reader_options=reader_options, harmonics=harmonics
        )
    elif path.endswith(tuple(extension_mapping["processed"].keys())):
        return lambda path: processed_file_reader(
            path, reader_options=reader_options, harmonics=harmonics
        )
    elif path.endswith(tuple(extension_mapping["raw"].keys())):
        return lambda path: raw_file_reader(
            path, reader_options=reader_options, harmonics=harmonics
        )
    else:
        show_error("File extension not supported.")
        return None


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
    filename, file_extension = _get_filename_extension(path)
    if file_extension == ".sdt":
        # Try reading .sdt with increasing 'index' numbers to collect all files as channels
        i = 0
        raw_data = []
        while True:
            try:
                _data = extension_mapping["raw"][".sdt"](path, {"index": i})
                raw_data.append(_data)
                i += 1
            except IndexError:
                break
        # Stack list of xarrays in a new axis "C" (shapes must match)
        for _data in raw_data:
            assert (
                _data.shape == raw_data[0].shape
            ), "Shapes from files in .sdt do not match!"
        raw_data = xr.concat(raw_data, dim="C")
    else:
        raw_data = extension_mapping["raw"][file_extension](
            path, reader_options
        )

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

    # Determine the number of steps for the progress bar
    if iter_axis is not None and iter_axis in raw_dims:
        iter_axis_index = raw_dims.index(iter_axis)
        n_steps = raw_data.shape[iter_axis_index]
    else:
        n_steps = len(harmonics) if isinstance(harmonics, list) else 1

    pbr = show_activity_progress(
        desc=f"Reading {filename}...", total=n_steps + 1
    )

    if iter_axis is None or iter_axis not in raw_dims:
        # Handle files without iteration axis or when keepdims=False squeezed it out
        if file_extension in [".tif", ".tiff"]:
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

        pbr.set_description("Computing phasor transform...")
        mean_intensity_image, G_image, S_image = phasor_from_signal(
            raw_data, axis=axis, harmonic=harmonics
        )
        pbr.update(n_steps)
        pbr.close()
        show_info(f"Loaded {filename}")
        channel_suffix = (
            " Intensity Image"
            if iter_axis is None
            else " Intensity Image: Channel 0"
        )
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
                "harmonics": harmonics,
            },
        }
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
            histogram_axis = channel_data.dims.index("H")

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

            mean_intensity_image, G_image, S_image = phasor_from_signal(
                channel_data,
                axis=histogram_axis,
                harmonic=harmonics,
            )
            add_kwargs = {
                "name": (
                    f"{filename} Intensity Image: Channel {channel_label}"
                ),
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
                    "harmonics": harmonics,
                },
            }
            layers.append((mean_intensity_image, add_kwargs))
        pbr.close()
        show_info(f"Loaded {filename}")
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

    # Read each file individually
    per_file_layers: list[list[tuple]] = []
    for p in paths:
        layers = raw_file_reader(
            p, reader_options=reader_options, harmonics=harmonics
        )
        per_file_layers.append(layers)

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
        integers, or None. Default is None, which sets all harmonics present
        in the file to be processed.

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
    if harmonics is None:
        harmonics = 'all'
    filename, file_extension = _get_filename_extension(path)
    pbr = show_activity_progress(desc=f"Loading {filename}...", total=3)
    reader_options = reader_options or {"harmonic": harmonics}
    mean_intensity_image, real, imag, attrs = extension_mapping["processed"][
        file_extension
    ](path, reader_options)
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
                    str(label).upper() for label in add_kwargs["axis_labels"]
                ]
                if 'Z' in labels:
                    z_idx = labels.index('Z')
            scale = [1.0] * mean_intensity_image.ndim
            scale[z_idx] = float(z_spacing_um)
            add_kwargs["scale"] = tuple(scale)
        except (ValueError, TypeError):
            pass

    layers.append((mean_intensity_image, add_kwargs))
    pbr.close()
    show_info(f"Loaded {filename}")
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
    for arg, value in args.items():
        if arg in sig.parameters:
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


def read_czi(path: str) -> xr.DataArray:
    """Read CZI file and return an xarray DataArray.

    Parameters
    ----------
    path : str
        Path to file.

    Returns
    -------
    data : xr.DataArray
        DataArray with dimensions named after the CZI axes (e.g., 'C', 'Z', 'Y', 'X').
    """
    with czifile.CziFile(path) as czi:
        data = czi.asxarray().squeeze()
    return data
