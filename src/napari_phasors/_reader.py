"""
This module contains functions to read files supported by `phasorpy.io`
and computes phasor coordinates with `phasorpy.phasor.phasor_from_signal`

"""

import inspect
import itertools
import json
import os
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np
import pandas as pd
import phasorpy.io as io
import tifffile
import xarray as xr
from napari.layers import Labels
from napari.utils.colormaps.colormap_utils import CYMRGB, MAGENTA_GREEN
from napari.utils.notifications import show_error
from phasorpy.phasor import phasor_from_signal

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
        # ".flif": lambda path: io.read_flif(path),
        # ".bh": lambda path: io.read_bh(path),
        # ".bhz": lambda path: io.read_bhz(path),
        # ".ifli": lambda path: io.read_ifli(),
    },
    "processed": {
        ".ome.tif": lambda path, reader_options: _parse_and_call_io_function(
            path, io.phasor_from_ometiff, {}, reader_options
        ),
        # ".b64": lambda path: io.read_b64(path),
        # ".r64": lambda path: io.read_r64(path),
        # ".ref": lambda path: io.read_ref(path)
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
    '.sdt': "C",
}
"""This dictionary contains the mapping for the axis to iterate over
when calculating phasor coordinates in the file.
"""


def napari_get_reader(
    path: str,
    reader_options: Optional[dict] = None,
    harmonics: Union[int, Sequence[int], None] = None,
) -> Optional[Callable]:
    """Initial reader function to map file extension to
    specific reader functions.

    Parameters
    ----------
    path : str
        Path to file.
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
    if path.endswith(tuple(extension_mapping["processed"].keys())):
        return lambda path: processed_file_reader(
            path, reader_options=reader_options, harmonics=harmonics
        )
    elif path.endswith(tuple(extension_mapping["raw"].keys())):
        return lambda path: raw_file_reader(
            path, reader_options=reader_options, harmonics=harmonics
        )
    else:
        show_error("File extension not supported.")


def raw_file_reader(
    path: str,
    reader_options: Optional[dict] = None,
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
        and 'frequency' in raw_data.attrs.keys()
    ):
        settings['frequency'] = raw_data.attrs['frequency']

    layers = []
    iter_axis = iter_index_mapping[file_extension]

    if iter_axis is None or iter_axis not in raw_data.dims:
        # Handle files without iteration axis or when keepdims=False squeezed it out
        if file_extension == ".tif":
            axis = 0
        elif iter_axis is None:
            # Hyperspectral files - find "C" dimension
            axis = raw_data.dims.index("C")
        else:
            # FLIM files without "C" dimension (single channel)
            axis = raw_data.dims.index("H")

        # Calculate summed signal over spatial dimensions
        if file_extension in [".lsm", ".tif"]:
            # For hyperspectral files, sum over spatial dimensions (axes 1 onwards)
            axes_to_sum = tuple(range(1, len(raw_data.shape)))
        else:
            # For FLIM files, sum over all dimensions except the histogram axis
            axes_to_sum = tuple(
                i for i in range(len(raw_data.shape)) if i != axis
            )

        summed_signal = np.sum(raw_data, axis=axes_to_sum)
        # Convert xarray DataArray to numpy array before converting to list
        if hasattr(summed_signal, 'values'):
            summed_signal = summed_signal.values
        settings['summed_signal'] = (
            summed_signal.tolist()
            if hasattr(summed_signal, 'tolist')
            else summed_signal
        )

        # Only set channel for files that actually have channels (FLIM files)
        if file_extension not in [".lsm", ".tif"]:
            settings['channel'] = 0

        mean_intensity_image, G_image, S_image = phasor_from_signal(
            raw_data, axis=axis, harmonic=harmonics
        )
        labels_layer = make_phasors_labels_layer(
            mean_intensity_image,
            G_image,
            S_image,
            name=filename,
            harmonics=harmonics,
        )
        channel_suffix = (
            " Intensity Image"
            if iter_axis is None
            else " Intensity Image: Channel 0"
        )
        add_kwargs = {
            "name": f"{filename}{channel_suffix}",
            "metadata": {
                "phasor_features_labels_layer": labels_layer,
                "original_mean": mean_intensity_image,
                "settings": settings,
            },
        }
        layers.append((mean_intensity_image, add_kwargs))
    else:
        # Handle multi-channel files with iteration axis
        iter_axis_index = raw_data.dims.index(iter_axis)
        for channel in range(raw_data.shape[iter_axis_index]):
            channel_data = raw_data.sel(C=channel)
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
            channel_settings['summed_signal'] = (
                summed_signal.tolist()
                if hasattr(summed_signal, 'tolist')
                else summed_signal
            )
            channel_settings['channel'] = channel

            mean_intensity_image, G_image, S_image = phasor_from_signal(
                channel_data,
                axis=histogram_axis,
                harmonic=harmonics,
            )
            labels_layer = make_phasors_labels_layer(
                mean_intensity_image,
                G_image,
                S_image,
                name=filename,
                harmonics=harmonics,
            )
            add_kwargs = {
                "name": f"{filename} Intensity Image: Channel {channel}",
                "metadata": {
                    "phasor_features_labels_layer": labels_layer,
                    "original_mean": mean_intensity_image,
                    "settings": channel_settings,
                },
            }
            layers.append((mean_intensity_image, add_kwargs))
    # Set colormaps if multichannel image
    if len(layers) == 2:
        # add colormaps MAGENTA_GREEN
        for layer, cmap in zip(layers, MAGENTA_GREEN):
            layer[1]["colormap"] = cmap
            layer[1]['blending'] = 'additive'
    elif len(layers) > 2:
        # add colormaps CYMRGB in a cycle
        for layer, cmap in zip(layers, itertools.cycle(CYMRGB)):
            layer[1]["colormap"] = cmap
            layer[1]['blending'] = 'additive'

    return layers


def processed_file_reader(
    path: str,
    reader_options: Optional[dict[str, str]] = None,
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
    reader_options = reader_options or {"harmonic": harmonics}
    mean_intensity_image, G_image, S_image, attrs = extension_mapping[
        "processed"
    ][file_extension](path, reader_options)
    if "description" in attrs.keys():
        description = json.loads(attrs["description"])
        if len(json.dumps(description)) > 512 * 512:  # Threshold: 256 KB
            raise ValueError("Description dictionary is too large.")
        if "napari_phasors_settings" in description:
            settings = json.loads(description["napari_phasors_settings"])
            if "calibrated" in settings.keys():
                settings["calibrated"] = bool(settings["calibrated"])
    else:
        settings = {}
    if "frequency" in attrs.keys():
        settings["frequency"] = attrs["frequency"]
    harmonics_read = attrs.get("harmonic", None)

    labels_layer = make_phasors_labels_layer(
        mean_intensity_image,
        G_image,
        S_image,
        name=filename,
        harmonics=harmonics_read,
    )

    original_mean_intensity_image = mean_intensity_image.copy()

    harmonics_array = np.unique(labels_layer.features['harmonic'])
    real = labels_layer.features['G_original'].copy()
    imag = labels_layer.features['S_original'].copy()
    real = np.reshape(
        real, (len(harmonics_array),) + mean_intensity_image.shape
    )
    imag = np.reshape(
        imag, (len(harmonics_array),) + mean_intensity_image.shape
    )

    should_apply_processing = False
    filter_params = {}
    threshold_value = 0

    if "filter" in settings.keys():
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

    if "threshold" in settings.keys() and settings["threshold"] is not None:
        should_apply_processing = True
        threshold_value = settings["threshold"]

    if should_apply_processing:
        from ._utils import _apply_filter_and_threshold_to_phasor_arrays

        mean_intensity_image, real, imag = (
            _apply_filter_and_threshold_to_phasor_arrays(
                mean_intensity_image,
                real,
                imag,
                harmonics_array,
                threshold=threshold_value,
                **filter_params,
            )
        )

        labels_layer.features['G'] = real.flatten()
        labels_layer.features['S'] = imag.flatten()

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

    layers = []
    add_kwargs = {
        "name": filename + " Intensity Image",
        "metadata": {
            "phasor_features_labels_layer": labels_layer,
            "original_mean": original_mean_intensity_image,
            "settings": settings,
        },
    }
    layers.append((mean_intensity_image, add_kwargs))
    return layers


def make_phasors_labels_layer(
    mean_intensity_image: Any,
    G_image: Any,
    S_image: Any,
    name: str = "",
    harmonics: Union[int, Sequence[int], None] = None,
) -> Labels:
    """Create a napari Labels layer from phasor coordinates.

    Parameters
    ----------
    mean_intensity_image : np.ndarray
        Mean intensity image.
    G_image : np.ndarray
        G phasor coordinates.
    S_image : np.ndarray
        S phasor coordinates.
    name : str, optional
        Name of the layer, by default ''.
    harmonics : Union[int, Sequence[int], None], optional
        Harmonic(s) to be processed. Can be a single integer, a sequence of
        integers, or None. Default is None.

    Returns
    -------
    labels_layer : napari.layers.Labels
        Labels layer with phasor coordinates as features.

    """
    pixel_id = np.arange(1, mean_intensity_image.size + 1)
    table = pd.DataFrame()
    if len(G_image.shape) > 2:
        for i in range(G_image.shape[0]):
            harmonic_value = harmonics[i] if harmonics is not None else i + 1
            sub_table = pd.DataFrame(
                {
                    "label": pixel_id,
                    "G_original": G_image[i].ravel(),
                    "S_original": S_image[i].ravel(),
                    "G": G_image[i].ravel(),
                    "S": S_image[i].ravel(),
                    "harmonic": harmonic_value,
                }
            )
            table = pd.concat([table, sub_table])
    else:
        if isinstance(harmonics, list):
            harmonic_value = harmonics[0]
        else:
            harmonic_value = harmonics if harmonics is not None else 1
        table = pd.DataFrame(
            {
                "label": pixel_id,
                "G_original": G_image.ravel(),
                "S_original": S_image.ravel(),
                "G": G_image.ravel(),
                "S": S_image.ravel(),
                "harmonic": harmonic_value,
            }
        )

    labels_data = pixel_id.reshape(mean_intensity_image.shape)
    labels_layer = Labels(
        labels_data,
        name=f"{name} Phasor Features Layer",
        scale=(1, 1),
        features=table,
    )
    return labels_layer


def _parse_and_call_io_function(
    path: str,
    func: Callable,
    args_defaults: dict[str, Any],
    reader_options: Optional[dict[str, Any]] = None,
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
    if len(parts) > 1:
        file_extension = "." + parts[1]
    else:
        file_extension = ""
    return parts[0], file_extension.lower()
