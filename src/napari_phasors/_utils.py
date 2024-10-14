"""
This module contains utility functions used by other modules.

"""

import warnings

import numpy as np
from napari.layers import Image
from phasorpy.phasor import phasor_filter, phasor_threshold


def apply_filter_and_threshold(
    layer: Image,
    /,
    *,
    threshold: float = 0,
    method: str = 'median',
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
        real, imag = phasor_filter(
            real,
            imag,
            method=method,
            repeat=repeat,
            size=size,
            axes=tuple(range(1, real.ndim)),
        )
    mean, real, imag = phasor_threshold(mean, real, imag, threshold)
    (
        layer.metadata['phasor_features_labels_layer'].features['G'],
        layer.metadata['phasor_features_labels_layer'].features['S'],
    ) = (real.flatten(), imag.flatten())
    if len(harmonics) > 1:
        # TODO: remove this when `phasor_threshold` handles multiple harmonics.
        # Catch the RuntimeWarning for all-NaN slices
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            merged_mean = np.nanmax(mean, axis=0)
            merged_mean[np.isnan(np.nanmin(mean, axis=0))] = np.nan
            mean = merged_mean
    layer.data = mean
    layer.refresh()
    return
