"""
This module contains utility functions used by other modules.

"""

import warnings
from typing import TYPE_CHECKING

import numpy as np
from napari.layers import Image
from phasorpy.phasor import (
    phasor_filter_median,
    phasor_filter_pawflim,
    phasor_threshold,
)

if TYPE_CHECKING:
    import napari


def validate_harmonics_for_wavelet(harmonics):
    """Validate that harmonics have their double or half correspondent.

    Parameters
    ----------
    harmonics : array-like
        Array of harmonic values

    Returns
    -------
    bool
        True if harmonics are valid for wavelet filtering, False otherwise
    """
    harmonics = np.array(harmonics)

    for harmonic in harmonics:
        # Check if double or half exists
        has_double = (harmonic * 2) in harmonics
        has_half = (harmonic / 2) in harmonics

        if not (has_double or has_half):
            return False

    return True


def _extract_phasor_arrays_from_layer(
    layer: Image, harmonics: np.ndarray = None
):
    """Extract phasor arrays from layer metadata.

    Parameters
    ----------
    layer : napari.layers.Image
        Napari image layer with phasor features.
    harmonics : np.ndarray, optional
        Harmonic values. If None, will be extracted from layer.

    Returns
    -------
    tuple
        (mean, real, imag, harmonics) arrays
    """
    mean = layer.metadata['original_mean'].copy()
    phasor_features = layer.metadata['phasor_features_labels_layer'].features

    if harmonics is None:
        harmonics = np.unique(phasor_features['harmonic'])

    real = phasor_features['G_original'].copy()
    imag = phasor_features['S_original'].copy()
    real = np.reshape(real, (len(harmonics),) + mean.shape)
    imag = np.reshape(imag, (len(harmonics),) + mean.shape)

    return mean, real, imag, harmonics


def _apply_filter_and_threshold_to_phasor_arrays(
    mean: np.ndarray,
    real: np.ndarray,
    imag: np.ndarray,
    harmonics: np.ndarray,
    *,
    threshold: float = 0,
    filter_method: str = "median",
    size: int = 3,
    repeat: int = 1,
    sigma: float = 1.0,
    levels: int = 3,
):
    """Apply filter and threshold to phasor arrays.

    Parameters
    ----------
    mean : np.ndarray
        Mean intensity array.
    real : np.ndarray
        Real part of phasor (G).
    imag : np.ndarray
        Imaginary part of phasor (S).
    harmonics : np.ndarray
        Harmonic values.
    threshold : float
        Threshold value for the mean value to be applied to G and S.
    filter_method : str
        Filter method. Options are 'median' or 'wavelet'.
    size : int
        Size of the median filter.
    repeat : int
        Number of times to apply the median filter.
    sigma : float
        Sigma parameter for wavelet filter.
    levels : int
        Number of levels for wavelet filter.

    Returns
    -------
    tuple
        (mean, real, imag) filtered and thresholded arrays
    """
    if filter_method == "median" and repeat > 0:
        mean, real, imag = phasor_filter_median(
            mean,
            real,
            imag,
            repeat=repeat,
            size=size,
        )
    elif filter_method == "wavelet" and validate_harmonics_for_wavelet(
        harmonics
    ):
        mean, real, imag = phasor_filter_pawflim(
            mean,
            real,
            imag,
            sigma=sigma,
            levels=levels,
            harmonic=harmonics,
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean, real, imag = phasor_threshold(mean, real, imag, threshold)

    return mean, real, imag


def apply_filter_and_threshold(
    layer: Image,
    /,
    *,
    threshold: float = 0,
    filter_method: str = "median",
    size: int = 3,
    repeat: int = 1,
    sigma: float = 1.0,
    levels: int = 3,
    harmonics: np.ndarray = None,
):
    """Apply filter to an image layer.

    Parameters
    ----------
    layer : napari.layers.Image
        Napari image layer with phasor features.
    threshold : float
        Threshold value for the mean value to be applied to G and S.
    filter_method : str
        Filter method. Options are 'median' or 'wavelet'.
    size : int
        Size of the median filter.
    repeat : int
        Number of times to apply the median filter.
    sigma : float
        Sigma parameter for wavelet filter.
    levels : int
        Number of levels for wavelet filter.
    harmonics : np.ndarray, optional
        Harmonic values for wavelet filter. If None, will be extracted from layer.

    """
    # Extract phasor arrays from layer
    mean, real, imag, harmonics = _extract_phasor_arrays_from_layer(
        layer, harmonics
    )

    # Apply filter and threshold to phasor arrays
    mean, real, imag = _apply_filter_and_threshold_to_phasor_arrays(
        mean,
        real,
        imag,
        harmonics,
        threshold=threshold,
        filter_method=filter_method,
        size=size,
        repeat=repeat,
        sigma=sigma,
        levels=levels,
    )

    # Update layer data and metadata
    layer.metadata['phasor_features_labels_layer'].features[
        'G'
    ] = real.flatten()
    layer.metadata['phasor_features_labels_layer'].features[
        'S'
    ] = imag.flatten()
    layer.data = mean

    # Update the settings dictionary of the layer
    if "settings" not in layer.metadata:
        layer.metadata["settings"] = {}
    layer.metadata["settings"]["filter"] = {
        "method": filter_method,
        "size": size,
        "repeat": repeat,
        "sigma": sigma,
        "levels": levels,
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


def update_frequency_in_metadata(
    image_layer: "napari.layers.Image",
    frequency: float,
):
    """Update the frequency in the layer metadata."""
    if "settings" not in image_layer.metadata.keys():
        image_layer.metadata["settings"] = {}
    image_layer.metadata["settings"]["frequency"] = frequency
