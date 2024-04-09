"""
This module is an example of a barebones numpy reader plugin for napari.

It implements the Reader specification, but your plugin may choose to
implement multiple readers or even other plugin contributions. see:
https://napari.org/stable/plugins/guides.html?#readers
"""
import os
import numpy as np
import phasorpy.io as io

def napari_get_reader(path):
    """A basic implementation of a Reader contribution.

    """
    extension_mapping = {
        ".ptu": io.read_ptu,
        ".fbd": read_fbd,
        ".flif": io.read_flif,
        ".sdt": io.read_sdt,
        ".bh": io.read_bh,
        ".bhz": io.read_bhz,
        ".b64": io.read_b64,
        ".ifli": io.read_ifli,
        ".r64": io.read_r64,
        ".ref": io.read_ref,
        ".ometiff": io.read_ometiff_phasor,
        ".lsm": io.read_lsm,
    }
    print('ENTERING READER')
    _, file_extension = os.path.splitext(path)
    print(f"FILE EXTENSION: {file_extension}")
    file_extension = file_extension.lower()
    processing_function = extension_mapping.get(file_extension, reader_function)
    # otherwise we return the *function* that can read ``path``.
    return processing_function(path)

def reader_function(path):
    """Take a path or list of paths and return a list of LayerData tuples.

    Readers are expected to return data as a list of tuples, where each tuple
    is (data, [add_kwargs, [layer_type]]), "add_kwargs" and "layer_type" are
    both optional.

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
    # handle both a string and a list of strings
    paths = [path] if isinstance(path, str) else path
    # load all files into array
    arrays = [np.load(_path) for _path in paths]
    # stack arrays into single array
    data = np.squeeze(np.stack(arrays))

    # optional kwargs for the corresponding viewer.add_* method
    add_kwargs = {}

    layer_type = "image"  # optional, default is "image"
    return [(data, add_kwargs, layer_type)]

def read_fbd(path):
    data = io.read_fbd(path, frame=-1,keepdims=False)
    return data