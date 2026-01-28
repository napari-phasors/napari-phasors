"""
This module contains utility functions used by other modules.

"""

import warnings
from typing import TYPE_CHECKING

import numpy as np
from napari.layers import Image
from phasorpy.filter import (
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
    harmonics = np.atleast_1d(harmonics)

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

    if harmonics is None:
        harmonics = layer.metadata.get('harmonics')

    harmonics = np.atleast_1d(harmonics)

    real = layer.metadata['G_original'].copy()
    imag = layer.metadata['S_original'].copy()

    # Apply mask if present in metadata
    if 'mask' in layer.metadata:
        mask = layer.metadata['mask']
        # Apply mask: set values to NaN where mask <= 0
        mask_invalid = mask <= 0
        mean = np.where(mask_invalid, np.nan, mean)
        for h in range(len(harmonics)):
            real[h] = np.where(mask_invalid, np.nan, real[h])
            imag[h] = np.where(mask_invalid, np.nan, imag[h])

    return mean, real, imag, harmonics


def _apply_filter_and_threshold_to_phasor_arrays(
    mean: np.ndarray,
    real: np.ndarray,
    imag: np.ndarray,
    harmonics: np.ndarray,
    *,
    threshold: float = None,
    filter_method: str = None,
    size: int = None,
    repeat: int = None,
    sigma: float = None,
    levels: int = None,
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
    threshold : float, optional
        Threshold value for the mean value to be applied to G and S.
        If None, no threshold is applied.
    filter_method : str, optional
        Filter method. Options are 'median' or 'wavelet'.
        If None, no filter is applied.
    size : int, optional
        Size of the median filter. Only used if filter_method is 'median'.
    repeat : int, optional
        Number of times to apply the median filter. Only used if filter_method is 'median'.
    sigma : float, optional
        Sigma parameter for wavelet filter. Only used if filter_method is 'wavelet'.
    levels : int, optional
        Number of levels for wavelet filter. Only used if filter_method is 'wavelet'.

    Returns
    -------
    tuple
        (mean, real, imag) filtered and thresholded arrays
    """
    # Apply filter only if filter_method is specified
    if filter_method == "median" and repeat is not None and repeat > 0:
        mean, real, imag = phasor_filter_median(
            mean,
            real,
            imag,
            repeat=repeat,
            size=size if size is not None else 3,
        )
    elif filter_method == "wavelet" and validate_harmonics_for_wavelet(
        harmonics
    ):
        mean, real, imag = phasor_filter_pawflim(
            mean,
            real,
            imag,
            sigma=sigma if sigma is not None else 1.0,
            levels=levels if levels is not None else 3,
            harmonic=harmonics,
        )

    if threshold is not None and threshold > 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean, real, imag = phasor_threshold(mean, real, imag, threshold)

    return mean, real, imag


def apply_filter_and_threshold(
    layer: Image,
    /,
    *,
    threshold: float = None,
    threshold_method: str = None,
    filter_method: str = None,
    size: int = None,
    repeat: int = None,
    sigma: float = None,
    levels: int = None,
    harmonics: np.ndarray = None,
):
    """Apply filter to an image layer.

    Parameters
    ----------
    layer : napari.layers.Image
        Napari image layer with phasor features.
    threshold : float, optional
        Threshold value for the mean value to be applied to G and S.
        If None, no threshold is applied.
    threshold_method : str, optional
        Threshold method used. If None, no threshold method is saved.
    filter_method : str, optional
        Filter method. Options are 'median' or 'wavelet'.
        If None, no filter is applied.
    size : int, optional
        Size of the median filter. Only used if filter_method is 'median'.
    repeat : int, optional
        Number of times to apply the median filter. Only used if filter_method is 'median'.
    sigma : float, optional
        Sigma parameter for wavelet filter. Only used if filter_method is 'wavelet'.
    levels : int, optional
        Number of levels for wavelet filter. Only used if filter_method is 'wavelet'.
    harmonics : np.ndarray, optional
        Harmonic values for wavelet filter. If None, will be extracted from layer.

    """
    mean, real, imag, harmonics = _extract_phasor_arrays_from_layer(
        layer, harmonics
    )

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

    layer.metadata['G'] = real
    layer.metadata['S'] = imag
    layer.data = mean

    if "settings" not in layer.metadata:
        layer.metadata["settings"] = {}

    # Only save filter settings if a filter was actually applied
    if filter_method is not None:
        layer.metadata["settings"]["filter"] = {}
        layer.metadata["settings"]["filter"]["method"] = filter_method

        if filter_method == "median":
            if size is not None:
                layer.metadata["settings"]["filter"]["size"] = size
            if repeat is not None:
                layer.metadata["settings"]["filter"]["repeat"] = repeat
        elif filter_method == "wavelet":
            if sigma is not None:
                layer.metadata["settings"]["filter"]["sigma"] = sigma
            if levels is not None:
                layer.metadata["settings"]["filter"]["levels"] = levels

    # Only save threshold settings if a threshold was actually applied
    if threshold is not None and threshold > 0:
        layer.metadata["settings"]["threshold"] = threshold
    if threshold_method is not None and threshold_method != "None":
        layer.metadata["settings"]["threshold_method"] = threshold_method

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
