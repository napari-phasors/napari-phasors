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


def threshold_otsu(data, nbins=256):
    """Calculate Otsu's threshold for the given data.

    Otsu's method finds the threshold that minimizes the weighted
    within-class variance, which is equivalent to maximizing the
    between-class variance.

    Parameters
    ----------
    data : np.ndarray
        1D array of values (NaN values should be removed beforehand).
    nbins : int, optional
        Number of histogram bins. Default is 256.

    Returns
    -------
    float
        The optimal threshold value.

    References
    ----------
    .. [1] Otsu, N., "A Threshold Selection Method from Gray-Level
           Histograms", IEEE Transactions on Systems, Man, and
           Cybernetics, vol. 9, no. 1, pp. 62-66, 1979.
    """
    data = np.asarray(data, dtype=float).ravel()

    if data.size == 0:
        return 0.0

    min_val, max_val = data.min(), data.max()
    if min_val == max_val:
        return float(min_val)

    counts, bin_edges = np.histogram(
        data, bins=nbins, range=(min_val, max_val)
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # Class probabilities and means for all possible thresholds
    weight1 = np.cumsum(counts)
    weight2 = np.cumsum(counts[::-1])[::-1]

    # Avoid division by zero
    mean1 = np.cumsum(counts * bin_centers) / np.maximum(weight1, 1)
    mean2 = np.cumsum((counts * bin_centers)[::-1])[::-1] / np.maximum(
        weight2, 1
    )

    # Between-class variance
    variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    idx = np.argmax(variance)
    return float(bin_centers[idx])


def threshold_li(data, initial_guess=None, tolerance=None):
    """Calculate Li's minimum cross-entropy threshold.

    Li's iterative method finds the threshold that minimizes the
    cross-entropy between the foreground and background distributions.

    This implementation matches scikit-image's ``threshold_li`` by
    shifting the data so that the minimum is zero before iterating,
    then adding the shift back to the final threshold.

    Parameters
    ----------
    data : np.ndarray
        1D array of values (NaN values should be removed beforehand).
    initial_guess : float, optional
        Starting threshold. If None, uses the mean of the data.
    tolerance : float, optional
        Convergence tolerance. If None, uses half the smallest
        difference between unique values.

    Returns
    -------
    float
        The optimal threshold value.

    References
    ----------
    .. [1] Li, C.H. and Lee, C.K., "Minimum Cross Entropy Thresholding",
           Pattern Recognition, vol. 26, no. 4, pp. 617-625, 1993.
    .. [2] Li, C.H. and Tam, P.K.S., "An Iterative Algorithm for Minimum
           Cross Entropy Thresholding", Pattern Recognition Letters,
           vol. 19, no. 8, pp. 771-776, 1998.
    """
    data = np.asarray(data, dtype=float).ravel()

    if data.size == 0:
        return 0.0

    min_val, max_val = data.min(), data.max()
    if min_val == max_val:
        return float(min_val)

    # Shift data so that minimum is 0 (Li's method requires positive values
    # and the shift affects the threshold due to the logarithm).
    image_min = float(min_val)
    data = data - image_min

    if tolerance is None:
        sorted_unique = np.unique(data)
        if len(sorted_unique) > 1:
            diffs = np.diff(sorted_unique)
            tolerance = diffs[diffs > 0].min() / 2.0
        else:
            return float(min_val)

    # Initialise with the convention used by scikit-image:
    # t_next holds the *candidate* threshold, t_curr the *previous* one.
    if initial_guess is None:
        t_next = float(np.mean(data))
    else:
        t_next = float(initial_guess) - image_min

    t_curr = -2 * tolerance  # ensure first iteration always runs

    while abs(t_next - t_curr) > tolerance:
        t_curr = t_next
        foreground = data > t_curr
        mean_fore = np.mean(data[foreground])
        mean_back = np.mean(data[~foreground])

        if mean_back == 0.0:
            break

        t_next = (mean_back - mean_fore) / (
            np.log(mean_back) - np.log(mean_fore)
        )

    return float(t_next + image_min)


def threshold_yen(data, nbins=256):
    """Calculate Yen's threshold.

    Yen's method maximizes the correlation between the original and
    thresholded images in terms of their entropy.

    Parameters
    ----------
    data : np.ndarray
        1D array of values (NaN values should be removed beforehand).
    nbins : int, optional
        Number of histogram bins. Default is 256.

    Returns
    -------
    float
        The optimal threshold value.

    References
    ----------
    .. [1] Yen, J.C., Chang, F.J., and Chang, S., "A New Criterion for
           Automatic Multilevel Thresholding", IEEE Transactions on Image
           Processing, vol. 4, no. 3, pp. 370-378, 1995.
    """
    data = np.asarray(data, dtype=float).ravel()

    if data.size == 0:
        return 0.0

    min_val, max_val = data.min(), data.max()
    if min_val == max_val:
        return float(min_val)

    counts, bin_edges = np.histogram(
        data, bins=nbins, range=(min_val, max_val)
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # Normalize to probabilities
    total = counts.sum()
    if total == 0:
        return float(min_val)

    pmf = counts.astype(float) / total

    # Cumulative sums
    P1 = np.cumsum(pmf)  # P(class 1)
    P1_sq = np.cumsum(pmf**2)  # sum of p_i^2 for class 1
    P2_sq = np.cumsum(pmf[::-1] ** 2)[::-1]  # sum of p_j^2 for class 2

    # Yen's criterion (Eq. 4 in [1]):
    #   crit = log( (P1*(1-P1))^2 / (P1_sq * P2_sq) )
    crit = np.log(
        ((P1_sq[:-1] * P2_sq[1:]) ** -1) * (P1[:-1] * (1.0 - P1[:-1])) ** 2
    )

    idx = np.argmax(crit)
    return float(bin_centers[idx])


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
    threshold_upper: float = None,
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
        Lower threshold value for the mean value to be applied to G and S.
        If None, no lower threshold is applied.
    threshold_upper : float, optional
        Upper threshold value for the mean value to be applied to G and S.
        If None, no upper threshold is applied.
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

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean, real, imag = phasor_threshold(
            mean, real, imag, mean_min=threshold, mean_max=threshold_upper
        )

    return mean, real, imag


def apply_filter_and_threshold(
    layer: Image,
    /,
    *,
    threshold: float = None,
    threshold_upper: float = None,
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
        Lower threshold value for the mean value to be applied to G and S.
        If None, no lower threshold is applied.
    threshold_upper : float, optional
        Upper threshold value for the mean value to be applied to G and S.
        If None, no upper threshold is applied.
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
        threshold_upper=threshold_upper,
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

    layer.metadata["settings"]["threshold"] = threshold
    layer.metadata["settings"]["threshold_upper"] = threshold_upper
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
