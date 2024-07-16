"""
This module contains functions to read files supported by `phasorpy.io`
and computes phasor coordinates with `phasorpy.phasor.phasor_from_signal`

"""
import os
import numpy as np
import phasorpy.io as io
from phasorpy.phasor import phasor_from_signal
import pandas as pd
from napari.layers import Labels
import inspect

extension_mapping = {
    'raw': {
        ".ptu": lambda path, options: _parse_and_call_io_function(path, io.read_ptu, {"frame": (-1, False), "keepdims": (False, False)}, options),
        ".fbd": lambda path, options: _parse_and_call_io_function(path, io.read_fbd, {"frame": (-1, False), "keepdims": (False, False)}, options),
        ".lsm": lambda path, options: _parse_and_call_io_function(path, io.read_lsm, {}, options),
        # ".flif": lambda path: io.read_flif(path),
        # ".sdt": lambda path: io.read_sdt(path),
        # ".bh": lambda path: io.read_bh(path),
        # ".bhz": lambda path: io.read_bhz(path),
        # ".ifli": lambda path: io.read_ifli(),
    },
    'processed': {
        ".tif": lambda path, options: _parse_and_call_io_function(path, io.read_ometiff_phasor, {}, options),
        # ".b64": lambda path: io.read_b64(path),
        # ".r64": lambda path: io.read_r64(path),
        # ".ref": lambda path: io.read_ref(path)
    }
}
"""This dictionary contains the mapping for reader functions from
`phasorpy.io` supported formats.

Commented file extensions are not supported at the moment.

"""

iter_index_mapping = {
    ".ptu": 'C',
    ".lsm": None,
    ".fbd": 'C'
}
"""This dictionary contains the mapping for the axis to iterate over
when calculating phasor coordinates in the file.
"""

def napari_get_reader(path: str = None, options: dict = None):
    """Initial reader function to map file extension to
    specific reader functions.
    
    Parameters
    ----------
    path : str
        Path to file.
        
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
    _, file_extension = os.path.splitext(path)
    file_extension = file_extension.lower()
    if file_extension in extension_mapping['processed'].keys():
        return lambda  path: processed_file_reader(path, options=options)
    elif file_extension in extension_mapping['raw'].keys():
        return lambda path: raw_file_reader(path, options=options)
    else:
        return unkonwn_reader_function

def unkonwn_reader_function(path: str | list[str]):
    """Generalized reader function used for formats not supported.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples where each tuple in the list contains
        (data, metadata, layer_type), where data is a numpy array, metadata is
        a dict of keyword arguments for the corresponding viewer.add_* method
        in napari, and layer_type is a lower-case string naming the type of
        layer. Both "meta", and "layer_type" are optional. napari will
        default to layer_type=="image" if not provided

    """
    paths = [path] if isinstance(path, str) else path
    arrays = [np.load(_path) for _path in paths]
    data = np.squeeze(np.stack(arrays))
    add_kwargs = {}
    layer_type = "image"
    return [(data[0], add_kwargs, layer_type)]


def raw_file_reader(path: str = None, options = None):
    """Read raw data files from supported file formats and apply the phasor
    transformation to get mean intensity image and phasor coordinates.

    Parameters
    ----------
    path : str
        Path to file.

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
    raw_data = extension_mapping['raw'][file_extension](path, options)
    layers = []
    iter_axis = iter_index_mapping[file_extension]
    if iter_axis is None:
        # Calculate phasor over channels if file is of hyperspectral type
        mean_intensity_image, G_image, S_image = phasor_from_signal(raw_data, axis=raw_data.dims.index('C'))
        labels_layer = make_phasors_labels_layer(mean_intensity_image, G_image, S_image, name=filename)
        add_kwargs = {'name': f'{filename} Intensity Image', 'metadata':{'phasor_features_labels_layer': labels_layer}}
        layers.append((mean_intensity_image, add_kwargs))
    else:
        iter_axis_index = raw_data.dims.index(iter_axis)
        for channel in range(raw_data.shape[iter_axis_index]):
            # Calculate phasor over photon counts dimension if file is of FLIM type
            mean_intensity_image, G_image, S_image = phasor_from_signal(raw_data.sel(C=channel), axis=raw_data.sel(C=channel).dims.index('H'))
            labels_layer = make_phasors_labels_layer(mean_intensity_image, G_image, S_image, name=filename)
            add_kwargs = {'name': f'{filename} Intensity Image: Channel {channel}', 'metadata':{'phasor_features_labels_layer': labels_layer}}
            layers.append((mean_intensity_image, add_kwargs))
    return layers

def processed_file_reader(path: str = None, options: dict = None):
    """Reader function for files that contain processed images, as phasor
    coordinates or intensity images.

    Parameters
    ----------
    path : str
        Path to file.

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
    mean_intensity_image, G_image, S_image = extension_mapping['processed'][file_extension](path)
    mean_intensity_image, G_image, S_image = mean_intensity_image.values, G_image.values, S_image.values
    labels_layer = make_phasors_labels_layer(mean_intensity_image, G_image, S_image, name=filename)
    layers = []
    add_kwargs = {'name': filename + ' Intensity Image', 'metadata':{'phasor_features_labels_layer': labels_layer}}
    layers.append((mean_intensity_image, add_kwargs))
    return layers

def make_phasors_labels_layer(mean_intensity_image, G_image, S_image, name=''):
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

    Returns
    -------
    labels_layer : napari.layers.Labels
        Labels layer with phasor coordinates as features.

    """
    pixel_id = np.arange(1, mean_intensity_image.size + 1)
    if len(G_image.shape) > 2:
        table = pd.DataFrame([])
        for i in range(G_image.shape[0]):
            sub_table = pd.DataFrame({'label': pixel_id, 'G': G_image[i].ravel(), 'S': S_image[i].ravel(), 'harmonic': i+1})  
            table = pd.concat([table, sub_table])
    else:
        table = pd.DataFrame({'label': pixel_id, 'G': G_image.ravel(), 'S': S_image.ravel(), 'harmonic': 1})
    labels_data = pixel_id.reshape(mean_intensity_image.shape)
    labels_layer = Labels(labels_data, name=name + ' Phasor Features Layer', scale=(1, 1), features=table)
    return labels_layer


def _parse_and_call_io_function(path, func, args_defaults, options=None):
    """
    A private helper function to parse arguments and call the specified io function.

    Parameters:
    - path (str): The file path to be passed to the io function.
    - func (callable): The io function to be called.
    - args_defaults (dict): A dictionary where keys are argument names and values are tuples of
                            (default_value, is_required). If is_required is True and the argument
                            is not present in the args_defaults, an exception is raised.
    - options (dict, optional): A dictionary of arguments that may override the defaults and
                                provide additional arguments not specified in args_defaults.

    Returns:
    The result of calling the io function with the parsed arguments.
    """
    args = {}
    # Use options if provided, otherwise use the default
    if options is not None:
        for arg, value in options.items():
            args[arg] = value

    # Fill in defaults for any missing arguments not provided in options
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
            raise ValueError(f"Invalid argument '{arg}' for function {func.__name__}.")
    print(valid_args)
    return func(path, **valid_args)

def _get_filename_extension(path: str):
    """Get the filename and extension from a path."""
    filename = os.path.basename(path)
    _, file_extension = os.path.splitext(filename)
    return filename, file_extension.lower()
