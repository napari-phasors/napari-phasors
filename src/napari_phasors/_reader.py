"""
This module is an example of a barebones numpy reader plugin for napari.

It implements the Reader specification, but your plugin may choose to
implement multiple readers or even other plugin contributions. see:
https://napari.org/stable/plugins/guides.html?#readers
"""
import os
import numpy as np
import phasorpy.io as io
from phasorpy.phasor import phasor_from_signal

def napari_get_reader(path):
    """A basic implementation of a Reader contribution.

    """
    extension_mapping = {
        # ".ptu": ptu_reader,
        ".fbd": fbd_reader,
        # ".flif": flif_reader,
        # ".sdt": sdt_reader,
        # ".bh": bh_reader,
        # ".bhz": bhz_reader,
        # ".b64": b64_reader,
        # ".ifli": ifli_reader,
        # ".r64": r64_reader,
        # ".ref": ref_reader,
        # ".ometiff": ometiff_phasor_reader,
        # ".lsm": lsm_reader,
    }
    _, file_extension = os.path.splitext(path)
    file_extension = file_extension.lower()
    processing_function = extension_mapping.get(file_extension, reader_function)
    return processing_function



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
    return [(data[0], add_kwargs, layer_type)]


def fbd_reader(path):
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

    data = io.read_fbd(path, frame=-1, channel=0,keepdims=False)
    phasor = phasor_from_signal(data)
    # optional kwargs for the corresponding viewer.add_* method
    add_kwargs = {}

    layer_type = "image"  # optional, default is "image"
    return [(phasor[0], add_kwargs, layer_type),(phasor[1], add_kwargs, layer_type),(phasor[2], add_kwargs, layer_type)]