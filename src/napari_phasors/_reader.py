"""
This module contains functions to read files supported by `phasorpy.io`
and computes phasor coordinates with `phasorpy.phasor.phasor_from_signal`

"""

import inspect
import itertools
import json
import os
import sys
import warnings
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np
import pandas as pd
import phasorpy.io as io
import tifffile
from pathlib import Path
from napari.layers import Labels
from napari.utils.colormaps.colormap_utils import CYMRGB, MAGENTA_GREEN
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
            io.signal_from_ptu,
            {"frame": (-1, False), "keepdims": (True, False)},
            reader_options,
        ),
        ".fbd": lambda path, reader_options: _parse_and_call_io_function(
            path,
            io.signal_from_fbd,
            {"frame": (-1, False), "keepdims": (True, False)},
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
    path = Path(path) # Convert to Path object for easier manipulation
    if path.is_dir():
        return lambda path: stack_reader(
            Path(path), reader_options=reader_options, harmonics=harmonics
        )
    else:
        suffix = ''.join(path.suffixes) # Get the full suffix (e.g. .ome.tif)
        if suffix in tuple(extension_mapping["processed"].keys()):
            return lambda path: processed_file_reader(
                path, reader_options=reader_options, harmonics=harmonics
            )
        elif suffix in tuple(extension_mapping["raw"].keys()):
            return lambda path: raw_file_reader(
                path, reader_options=reader_options, harmonics=harmonics
            )
        else:
            show_error("File extension not supported.")

def _stack_sdt_channels(path):
    """Read .sdt files with increasing 'index' numbers and stack them as channels.

    Parameters
    ----------
    path : str
        Path to file.
    
    Returns
    -------
    raw_data : xarray.DataArray
        Stacked xarray DataArray with all channels.
    """
    from xarray import concat
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
    return concat(raw_data, dim="C")

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
    if file_extension == ".sdt":
        raw_data = _stack_sdt_channels(path)
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
    if iter_axis is None:
        if file_extension == ".tif":
            mean_intensity_image, G_image, S_image = phasor_from_signal(
                raw_data, axis=0, harmonic=harmonics
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

def extract_value_after_prefix(path, prefix='t'):
    """
    Extracts an the integer value from the file path based on the given prefix.
    For example, if prefix is 't', it will search for the pattern '_t<number>'.
    Strips the 0s from the beginning of the number.
    Returns the matching string with leading 0s stripped if found, otherwise an empty string.

    Parameters
    ----------
    path : Path
        Path object of the file.
    prefix : str
        Prefix to search for in the file name. Default is 't'.

    Returns
    -------
    str
        Extracted value after the prefix, with leading 0s stripped.
        If no match is found, returns an empty string.
    """
    import re
    regex = r'_' + prefix + r'(\d+)'
    match = re.search(regex, path.stem)
    return match.group(1).lstrip('0') if match else ''

def extract_paths_with_same_value_after_prefix(file_paths, extract_prefix='t', sort_prefix='z'):
    """
    Extracts all file paths with the same value after the given prefix.
    Returns a dictionary where the key is the extract prefix + value and the value is a list of paths sorted by the sort prefix.
    For example, if prefix is 't' and sort prefix is 'z', it will search for the pattern '_t<number>' and sort by '_z<number>'.

    Parameters
    ----------
    file_paths : list of Path
        List of Path objects of the files.
    extract_prefix : str
        Prefix to search for in the file name. Default is 't'.
    sort_prefix : str
        Prefix to sort the file paths by. Default is 'z'.

    Returns
    -------
    dict
        Dictionary where the key is the extract prefix + value and the value is a list of paths sorted by the sort prefix.
    """
    from natsort import natsorted
    file_paths = natsorted(file_paths, key=lambda x: extract_value_after_prefix(x, prefix=extract_prefix))
    paths_dict = {}
    for path in file_paths:
        value = extract_value_after_prefix(path, extract_prefix)
        if (extract_prefix+value) not in paths_dict:
            paths_dict[extract_prefix+value] = []
        paths_dict[extract_prefix+value].append(path)

    # Sort the paths in each list
    for k in paths_dict:
        paths_dict[k] = natsorted(paths_dict[k], key=lambda x: extract_value_after_prefix(x, sort_prefix))
    return paths_dict

def stack_reader(
        path: Path,
        reader_options: Optional[dict] = None,
        harmonics: Union[int, Sequence[int], None] = None,
) -> Optional[Callable]:
    file_paths, suffixes = zip(*[(p,''.join(p.suffixes)) for p in path.iterdir() if p.is_file()])
    most_frequent_file_extension = max(set(suffixes), key=suffixes.count)
    # Filter out files that do not have the most frequent extension
    file_paths = [p for p, s in zip(file_paths, suffixes) if s == most_frequent_file_extension]
    if most_frequent_file_extension in tuple(extension_mapping["processed"].keys()):
        print("To be implemented")
        return
            # return lambda path: processed_stack_reader(
            #     file_paths, most_frequent_file_extension, reader_options=reader_options, harmonics=harmonics
            # )
    elif most_frequent_file_extension in tuple(extension_mapping["raw"].keys()):
        return raw_stack_reader(
            file_paths=file_paths, file_extension=most_frequent_file_extension, reader_options=reader_options, harmonics=harmonics
        )
    else:
        show_error("File extension not supported.")

def raw_stack_reader(
          file_paths: list[Path],
          file_extension: str,
            reader_options: Optional[dict] = None,
            harmonics: Union[int, Sequence[int], None] = None,
) -> tuple[np.ndarray, dict]:
    from tqdm import tqdm
    folder_name = file_paths[0].parent.name
    structured_file_path_dict = extract_paths_with_same_value_after_prefix(file_paths, extract_prefix='t', sort_prefix='z')

    progress_bar = tqdm(
        total=sum((len(v) for v in structured_file_path_dict.values())),
        desc="Reading stack",
        unit="files",
    )

    z_list_mean_intensity, z_list_G, z_list_S, t_list_mean_intensity, t_list_G, t_list_S = [], [], [], [], [], []
    for t, list_of_z_slice_paths in structured_file_path_dict.items():
        for z_slice_path in list_of_z_slice_paths:
            if file_extension == ".sdt":
                _raw_data = _stack_sdt_channels(z_slice_path)
            else:
                _raw_data = extension_mapping["raw"][file_extension](
                    z_slice_path, reader_options
                )
            iter_axis = iter_index_mapping[file_extension]
            if iter_axis is None:
                # Single channel analysis
                if file_extension == ".tif":
                    mean_intensity_image, G_image, S_image = phasor_from_signal(
                        _raw_data, axis=0, harmonic=harmonics
                    )
                else:
                    # Calculate phasor over channels if file is of hyperspectral type
                    mean_intensity_image, G_image, S_image = phasor_from_signal(
                        _raw_data, axis=_raw_data.dims.index("C"), harmonic=harmonics
                    )
                # Add unitary axis representing channels
                mean_intensity_image = np.expand_dims(mean_intensity_image, axis=-1)
                G_image = np.expand_dims(G_image, axis=-1)
                S_image = np.expand_dims(S_image, axis=-1)
            else:
                iter_axis_index = _raw_data.dims.index(iter_axis)
                c_list_mean_intensity, c_list_G, c_list_S = [], [], []
                for channel in range(_raw_data.shape[iter_axis_index]):
                    # Calculate phasor over photon counts dimension if file is FLIM
                    mean_intensity_image, G_image, S_image = phasor_from_signal(
                        _raw_data.sel(C=channel),
                        axis=_raw_data.sel(C=channel).dims.index("H"),
                        harmonic=harmonics,
                    )
                    c_list_mean_intensity.append(mean_intensity_image)
                    c_list_G.append(G_image)
                    c_list_S.append(S_image)
                # Stack list having channel being last axis
                mean_intensity_image = np.stack(c_list_mean_intensity, axis=-1)
                G_image = np.stack(c_list_G, axis=-1)
                S_image = np.stack(c_list_S, axis=-1)
            z_list_mean_intensity.append(mean_intensity_image)
            z_list_G.append(G_image)
            z_list_S.append(S_image)
            progress_bar.update(1)
        z_stack_mean_intensity = np.stack(z_list_mean_intensity, axis=1)
        z_stack_G = np.stack(z_list_G)
        z_stack_S = np.stack(z_list_S)
        t_list_mean_intensity.append(z_stack_mean_intensity)
        t_list_G.append(z_stack_G)
        t_list_S.append(z_stack_S)
        z_list_mean_intensity.clear()
        z_list_G.clear()
        z_list_S.clear()
    stack_mean_intensity = np.stack(t_list_mean_intensity, axis=1) # TZYXC
    stack_G = np.stack(t_list_G) # QTZYXC (Q for number of harmonics)
    stack_S = np.stack(t_list_S)
    progress_bar.close()
    settings = {}
    if (
        file_extension != '.fbd'
        and hasattr(_raw_data, "attrs")
        and 'frequency' in _raw_data.attrs.keys()
    ):
        settings['frequency'] = _raw_data.attrs['frequency']
    layers = []
    for ch in range(stack_mean_intensity.shape[-1]):
        labels_layer = make_phasors_labels_layer(
                stack_mean_intensity[..., ch],
                stack_G[..., ch],
                stack_S[..., ch],
                name=folder_name,
                harmonics=harmonics,
            )
        layer_name = [f"{folder_name} Intensity Image: " if stack_mean_intensity.shape[-1] == 1 else f"{folder_name} Intensity Image: Channel {ch}"][0]
        add_kwargs = {
            "name": layer_name,
            "metadata": {
                "phasor_features_labels_layer": labels_layer,
                "original_mean": stack_mean_intensity[..., ch],
                "settings": settings,
            },
        }
        layers.append((stack_mean_intensity[..., ch], add_kwargs))
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
    if len(G_image.shape) > len(mean_intensity_image.shape):
        # If G_image has more dimensions than mean_intensity_image, it means it has multiple harmonics in the first axis
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
        scale=tuple(np.ones(mean_intensity_image.ndim)), #TODO: apply proper scale based on metadata
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
