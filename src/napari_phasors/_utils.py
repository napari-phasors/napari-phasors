"""
This module contains utility functions used by other modules.

"""

import warnings

import numpy as np
from napari.layers import Image
from phasorpy.phasor import phasor_filter_median, phasor_threshold


def apply_filter_and_threshold(
    layer: Image,
    /,
    *,
    threshold: float = 0,
    size: int = 3,
    repeat: int = 1,
):
    """Apply filter to an image layer.

    Parameters
    ----------
    layer : napari.layers.Image
        Napari image layer with phasor features.
    threshold : float
        Threshold value for the mean value to be applied to G and S.
    method : str
        Filter method. Options are 'median'.
    size : int
        Size of the filter.
    repeat : int
        Number of times to apply the filter.

    """
    mean = layer.metadata['original_mean'].copy()
    phasor_features = layer.metadata['phasor_features_labels_layer'].features
    harmonics = np.unique(phasor_features['harmonic'])
    real, imag = (
        phasor_features['G_original'].copy(),
        phasor_features['S_original'].copy(),
    )
    real = np.reshape(real, (len(harmonics),) + mean.shape)
    imag = np.reshape(imag, (len(harmonics),) + mean.shape)
    if repeat > 0:
        mean, real, imag = phasor_filter_median(
            mean,
            real,
            imag,
            repeat=repeat,
            size=size,
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean, real, imag = phasor_threshold(mean, real, imag, threshold)
        (
            layer.metadata['phasor_features_labels_layer'].features['G'],
            layer.metadata['phasor_features_labels_layer'].features['S'],
        ) = (real.flatten(), imag.flatten())
    layer.data = mean
    # Update the settings dictionary of the layer
    if "settings" not in layer.metadata:
        layer.metadata["settings"] = {}
    layer.metadata["settings"]["filter"] = {
        "size": size,
        "repeat": repeat,
    }
    layer.metadata["settings"]["threshold"] = threshold
    layer.refresh()
    return


def colormap_to_dict(colormap, num_colors=10, exclude_first=True):
    """
    Converts a matplotlib colormap into a dictionary of RGBA colors.

    Parameters
    ----------
    colormap : matplotlib.colors.Colormap
        The colormap to convert.
    num_colors : int, optional
        The number of colors in the colormap, by default 10.
    exclude_first : bool, optional
        Whether to exclude the first color in the colormap, by default True.

    Returns
    -------
    color_dict: dict
        A dictionary with keys as positive integers and values as RGBA colors.
    """
    color_dict = {}
    start = 0
    if exclude_first:
        start = 1
    for i in range(start, num_colors + start):
        pos = i / (num_colors - 1)
        color = colormap(pos)
        color_dict[i + 1 - start] = color
    color_dict[None] = (0, 0, 0, 0)
    return color_dict
