"""
This module contains functions to read files supported by `phasorpy.io`
and computes phasor coordinates with `phasorpy.phasor.phasor_from_signal`

"""

import inspect
import json
import os
import sys
import warnings
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np
import pandas as pd
import phasorpy.io as io
import tifffile
from napari.layers import Labels
from napari.utils.notifications import show_error
from phasorpy.phasor import (
    phasor_filter_median,
    phasor_from_signal,
    phasor_threshold,
)

extension_mapping = {
    "raw": {
        ".ptu": lambda path, reader_options: _parse_and_call_io_function(
            path,
            io.read_ptu,
            {"frame": (-1, False), "keepdims": (True, False)},
            reader_options,
        ),
        ".fbd": lambda path, reader_options: _parse_and_call_io_function(
            path,
            io.read_fbd,
            {"frame": (-1, False), "keepdims": (True, False)},
            reader_options,
        ),
        ".sdt": lambda path, reader_options: _parse_and_call_io_function(
            path,
            io.read_sdt,
            {},
            reader_options,
        ),
        ".lsm": lambda path, reader_options: _parse_and_call_io_function(
            path,
            io.read_lsm,
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
    '.sdt': None,
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
        integers, or None. Default is None.

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
    filename, file_extension = _get_filename_extension(path)
    raw_data = extension_mapping["raw"][file_extension](path, reader_options)
    settings = {}
    if (
        file_extension != '.fbd'
        and hasattr(raw_data, "attrs")
        and 'frequency' in raw_data.attrs.keys()
    ):
        settings['frequency'] = raw_data.attrs['frequency']
    layers = []
    iter_axis = iter_index_mapping[file_extension]
    if iter_axis is None:
        if file_extension == ".tif":
            mean_intensity_image, G_image, S_image = phasor_from_signal(
                raw_data, axis=0, harmonic=harmonics
            )
        elif file_extension == '.sdt':
            mean_intensity_image, G_image, S_image = phasor_from_signal(
                raw_data, axis=-1, harmonic=harmonics
            )
        else:
            # Calculate phasor over channels if file is of hyperspectral type
            mean_intensity_image, G_image, S_image = phasor_from_signal(
                raw_data, axis=raw_data.dims.index("C"), harmonic=harmonics
            )
        labels_layer = make_phasors_labels_layer(
            mean_intensity_image,
            G_image,
            S_image,
            name=filename,
            harmonics=harmonics,
        )
        add_kwargs = {
            "name": f"{filename} Intensity Image",
            "metadata": {
                "phasor_features_labels_layer": labels_layer,
                "original_mean": mean_intensity_image,
                "settings": settings,
            },
        }
        layers.append((mean_intensity_image, add_kwargs))
    else:
        iter_axis_index = raw_data.dims.index(iter_axis)
        for channel in range(raw_data.shape[iter_axis_index]):
            # Calculate phasor over photon counts dimension if file is FLIM
            mean_intensity_image, G_image, S_image = phasor_from_signal(
                raw_data.sel(C=channel),
                axis=raw_data.sel(C=channel).dims.index("H"),
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
                    "settings": settings,
                },
            }
            layers.append((mean_intensity_image, add_kwargs))
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
        integers, or None. Default is None.

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
    filename, file_extension = _get_filename_extension(path)
    reader_options = reader_options or {"harmonic": harmonics}
    mean_intensity_image, G_image, S_image, attrs = extension_mapping[
        "processed"
    ][file_extension](path, reader_options)
    if "description" in attrs.keys():
        description = json.loads(attrs["description"])
        if sys.getsizeof(description) > 512 * 512:  # Threshold: 256 KB
            raise ValueError("Description dictionary is too large.")
        if "napari_phasors_settings" in description:
            settings = json.loads(description["napari_phasors_settings"])
            if "calibrated" in settings.keys():
                settings["calibrated"] = bool(settings["calibrated"])
    else:
        settings = {}
    if "frequency" in attrs.keys():
        settings["frequency"] = attrs["frequency"]
    labels_layer = make_phasors_labels_layer(
        mean_intensity_image,
        G_image,
        S_image,
        name=filename,
        harmonics=harmonics,
    )

    filter_size = None
    filter_repeat = None
    if "filter" in settings.keys():
        filter_size = settings["filter"]["size"]
        filter_repeat = settings["filter"]["repeat"]
    threshold = None
    if "threshold" in settings.keys():
        threshold = settings["threshold"]

    if threshold is not None or filter_repeat is not None:
        harmonics = np.unique(labels_layer.features['harmonic'])
        real, imag = (
            labels_layer.features['G_original'].copy(),
            labels_layer.features['S_original'].copy(),
        )
        mean = mean_intensity_image
        real = np.reshape(real, (len(harmonics),) + mean.shape)
        imag = np.reshape(imag, (len(harmonics),) + mean.shape)
        if filter_repeat is not None and filter_repeat > 0:
            if filter_size is None:
                filter_size = 3
            mean, real, imag = phasor_filter_median(
                mean,
                real,
                imag,
                repeat=filter_repeat,
                size=filter_size,
            )
        if threshold is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                mean, real, imag = phasor_threshold(
                    mean, real, imag, threshold
                )
                (
                    labels_layer.features['G'],
                    labels_layer.features['S'],
                ) = (real.flatten(), imag.flatten())
            mean_intensity_image = mean

    layers = []
    add_kwargs = {
        "name": filename + " Intensity Image",
        "metadata": {
            "phasor_features_labels_layer": labels_layer,
            "original_mean": mean_intensity_image,
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
