"""Spatial binning of phasor coordinates, and the level-of-detail cache.

Binning phasor data is not the same as downsampling an image. The phasor of a
sum of signals is the *photon-weighted* average of the individual phasors, so
the only correct way to shrink ``G``/``S`` is

.. math::

    G_\\mathrm{bin} = \\frac{\\sum_i \\bar{I}_i G_i}{\\sum_i \\bar{I}_i}

which is exactly the rule :func:`~napari_phasors._stitching.blend_phasor_tiles`
already uses where tiles overlap. Applying it here means a binned level is
*identical* to what phasor-transforming the summed raw signal would have
produced — binning costs resolution, never correctness. A plain (unweighted)
mean of ``G`` would quietly bias every bin towards its dimmest pixels.

Intensity is reported as the arithmetic mean over each block rather than the
sum, so a binned level keeps the same intensity scale as full resolution and
thresholds, contrast limits and colormaps carry across levels unchanged.

Blocks are aligned to the origin and the array is padded, never trimmed, so
bin ``(i, j)`` always covers full-resolution rows ``[i*f, (i+1)*f)``. That
exact correspondence is what lets :class:`PhasorPyramid` refine an arbitrary
zoomed region and paste the answer back into the coarse canvas.
"""

import numpy as np

__all__ = [
    "bin_phasor_arrays",
    "bin_factor_for_shape",
    "binned_shape",
    "PhasorPyramid",
]

#: Above this many pixels an image is treated as "large" and binned on load.
#: Roughly the point where a phasor plot's 2D histogram and the per-layer
#: filters stop feeling interactive on a typical laptop.
DEFAULT_PIXEL_BUDGET = 4_000_000

#: Never bin harder than this, however big the image.
MAX_BIN_FACTOR = 32


def binned_shape(shape, factor):
    """Return the shape *shape* takes after binning its last two axes."""
    factor = max(1, int(factor))
    if factor == 1:
        return tuple(shape)
    *lead, height, width = shape
    return (
        *lead,
        -(-int(height) // factor),  # ceil, because blocks are padded
        -(-int(width) // factor),
    )


def bin_factor_for_shape(shape, budget=DEFAULT_PIXEL_BUDGET):
    """Return the smallest bin factor bringing *shape* under *budget* pixels.

    Parameters
    ----------
    shape : tuple of int
        Image shape. Only the last two (spatial) axes are binned, but every
        axis counts towards the pixel budget, so a 40-plane stack bins more
        aggressively than a single plane of the same width.
    budget : int, optional
        Target pixel count.

    Returns
    -------
    int
        A power of two in ``[1, MAX_BIN_FACTOR]``. ``1`` means the image is
        already small enough to work with at full resolution.
    """
    shape = tuple(int(axis) for axis in shape)
    if len(shape) < 2 or budget <= 0:
        return 1

    total = int(np.prod(shape))
    factor = 1
    # Powers of two keep levels nested: every level is an exact re-bin of the
    # one above it, so refining never has to go back to full resolution.
    while factor < MAX_BIN_FACTOR:
        if int(np.prod(binned_shape(shape, factor))) <= budget:
            break
        factor *= 2
    else:
        return MAX_BIN_FACTOR
    return factor if total > budget else 1


def _pad_to_multiple(array, factor, fill):
    """Pad the last two axes of *array* up to a multiple of *factor*."""
    height, width = array.shape[-2:]
    pad_y = (-height) % factor
    pad_x = (-width) % factor
    if not pad_y and not pad_x:
        return array
    pad = [(0, 0)] * (array.ndim - 2) + [(0, pad_y), (0, pad_x)]
    return np.pad(array, pad, mode="constant", constant_values=fill)


def _block_reduce_sum(array, factor):
    """Sum the last two axes of *array* over ``factor`` x ``factor`` blocks."""
    *lead, height, width = array.shape
    reshaped = array.reshape(
        *lead, height // factor, factor, width // factor, factor
    )
    return reshaped.sum(axis=(-3, -1))


def bin_phasor_arrays(mean, real, imag, factor):
    """Bin phasor coordinates by *factor*, weighting by photon count.

    Parameters
    ----------
    mean : numpy.ndarray
        Mean intensity, shape ``(..., Y, X)``.
    real, imag : numpy.ndarray
        Phasor coordinates, either the same shape as *mean* or with a leading
        harmonic axis, ``(H, ..., Y, X)``.
    factor : int
        Block size. ``1`` returns the inputs untouched (not copied).

    Returns
    -------
    mean, real, imag : numpy.ndarray
        Binned arrays, ``float32``. Blocks with no valid pixel are ``NaN`` in
        every output, matching how the rest of the plugin marks "no data".

    Raises
    ------
    ValueError
        If *real* and *imag* disagree in shape, or their trailing spatial
        axes do not match *mean*.
    """
    factor = max(1, int(factor))
    if factor == 1:
        return mean, real, imag

    mean = np.asarray(mean)
    real = np.asarray(real)
    imag = np.asarray(imag)

    if real.shape != imag.shape:
        raise ValueError(
            f"real and imag disagree in shape: {real.shape} vs {imag.shape}."
        )
    if real.shape[-mean.ndim :] != mean.shape:
        raise ValueError(
            f"phasor coordinates of shape {real.shape} do not match a mean "
            f"of shape {mean.shape}."
        )

    stacked = real.ndim == mean.ndim + 1
    real_work = real if stacked else real[np.newaxis]
    imag_work = imag if stacked else imag[np.newaxis]

    # A pixel contributes only where the intensity and every harmonic of both
    # coordinates are finite, mirroring blend_phasor_tiles. One shared weight
    # keeps all harmonics on the same footing, which matters for multi-
    # harmonic fits that assume a common support.
    valid = np.isfinite(mean)
    valid &= np.all(np.isfinite(real_work), axis=0)
    valid &= np.all(np.isfinite(imag_work), axis=0)

    # Negative intensities (possible after background subtraction) would flip
    # the convex combination, so they carry no weight.
    weight = np.where(valid, np.clip(mean, 0.0, None), 0.0).astype(np.float64)

    weight_padded = _pad_to_multiple(weight, factor, 0.0)
    valid_padded = _pad_to_multiple(valid, factor, False)
    mean_padded = _pad_to_multiple(
        np.where(valid, mean, 0.0).astype(np.float64), factor, 0.0
    )

    weight_sum = _block_reduce_sum(weight_padded, factor)
    valid_count = _block_reduce_sum(valid_padded.astype(np.float64), factor)
    mean_sum = _block_reduce_sum(mean_padded, factor)

    has_photons = weight_sum > 0
    safe_weight = np.where(has_photons, weight_sum, 1.0)

    out_real = np.empty(
        (real_work.shape[0], *weight_sum.shape), dtype=np.float64
    )
    out_imag = np.empty_like(out_real)
    for harmonic in range(real_work.shape[0]):
        numerator_real = _block_reduce_sum(
            _pad_to_multiple(
                weight * np.where(valid, real_work[harmonic], 0.0),
                factor,
                0.0,
            ),
            factor,
        )
        numerator_imag = _block_reduce_sum(
            _pad_to_multiple(
                weight * np.where(valid, imag_work[harmonic], 0.0),
                factor,
                0.0,
            ),
            factor,
        )
        out_real[harmonic] = np.where(
            has_photons, numerator_real / safe_weight, np.nan
        )
        out_imag[harmonic] = np.where(
            has_photons, numerator_imag / safe_weight, np.nan
        )

    # Intensity is averaged over the block's valid pixels, so the binned level
    # sits on the same scale as full resolution.
    out_mean = np.where(
        valid_count > 0,
        mean_sum / np.where(valid_count > 0, valid_count, 1.0),
        np.nan,
    )

    if not stacked:
        out_real = out_real[0]
        out_imag = out_imag[0]

    return (
        out_mean.astype(np.float32),
        out_real.astype(np.float32),
        out_imag.astype(np.float32),
    )


def _snap(value, factor, limit):
    """Snap *value* down to the block grid, clamped into ``[0, limit]``."""
    value = int(max(0, min(int(value), limit)))
    return value - value % factor


class PhasorPyramid:
    """Level-of-detail cache over one layer's full-resolution phasor data.

    The plugin already keeps pristine ``original_mean`` / ``G_original`` /
    ``S_original`` arrays in every layer's metadata, so a pyramid does not
    need to own a second copy of the data: it holds references and derives
    coarser levels on demand.

    Two kinds of view are served.

    ``level(factor)``
        The whole image at one bin factor. Used when zoomed out.

    ``region(factor, ...)``
        A sub-rectangle at a finer factor. Used when zoomed in, so the cost
        of extra detail is bounded by the size of the viewport rather than
        the size of the image.

    Region starts are snapped down onto the block grid, so a region's bins
    line up exactly with the corresponding bins of any coarser level. That
    keeps world coordinates stable as the viewer swaps between levels: a
    region placed at ``translate=origin`` with ``scale=factor`` lands exactly
    where the coarse level had it.

    Parameters
    ----------
    mean : numpy.ndarray
        Full-resolution mean intensity, shape ``(..., Y, X)``.
    real, imag : numpy.ndarray
        Full-resolution phasor coordinates.
    max_cached_levels : int, optional
        How many whole-image levels to retain. The most recently used are
        kept; full resolution is never cached (it is the source data).
    """

    def __init__(self, mean, real, imag, max_cached_levels=3):
        self.mean = mean
        self.real = real
        self.imag = imag
        self.max_cached_levels = max(1, int(max_cached_levels))
        self._levels = {}
        self._order = []

    @property
    def shape(self):
        """Full-resolution spatial shape, ``(Y, X)``."""
        return tuple(self.mean.shape[-2:])

    @property
    def full_shape(self):
        """Full-resolution shape of the mean array, including leading axes."""
        return tuple(self.mean.shape)

    def available_factors(self):
        """Return the bin factors currently held in the cache, coarsest last."""
        return sorted(self._levels)

    def clear(self):
        """Drop every cached level."""
        self._levels.clear()
        self._order.clear()

    def nbytes(self):
        """Return the number of bytes the cached levels occupy."""
        return sum(
            sum(array.nbytes for array in arrays)
            for arrays in self._levels.values()
        )

    def level(self, factor):
        """Return ``(mean, real, imag)`` for the whole image at *factor*.

        ``factor=1`` returns the source arrays themselves, uncached and
        uncopied.
        """
        factor = max(1, int(factor))
        if factor == 1:
            return self.mean, self.real, self.imag

        cached = self._levels.get(factor)
        if cached is not None:
            self._touch(factor)
            return cached

        arrays = bin_phasor_arrays(self.mean, self.real, self.imag, factor)
        self._levels[factor] = arrays
        self._touch(factor)
        self._evict()
        return arrays

    def _touch(self, factor):
        if factor in self._order:
            self._order.remove(factor)
        self._order.append(factor)

    def _evict(self):
        while len(self._order) > self.max_cached_levels:
            self._levels.pop(self._order.pop(0), None)

    def region_bounds(self, factor, row_start, row_stop, col_start, col_stop):
        """Snap a full-resolution rectangle onto the *factor* block grid.

        Returns
        -------
        tuple of int
            ``(row_start, row_stop, col_start, col_stop)``, with the starts
            snapped down to multiples of *factor* and the stops raised to
            cover the request, clipped to the image.
        """
        factor = max(1, int(factor))
        height, width = self.shape

        row_start = _snap(row_start, factor, height)
        col_start = _snap(col_start, factor, width)
        row_stop = int(max(row_start + factor, min(int(row_stop), height)))
        col_stop = int(max(col_start + factor, min(int(col_stop), width)))
        # Raise the stop to a whole number of blocks so the region's last bin
        # is not a partial one that would disagree with the coarse grid.
        row_stop = min(
            height, row_start + -(-(row_stop - row_start) // factor) * factor
        )
        col_stop = min(
            width, col_start + -(-(col_stop - col_start) // factor) * factor
        )
        return row_start, row_stop, col_start, col_stop

    def region(self, factor, row_start, row_stop, col_start, col_stop):
        """Return one rectangle of the image, binned by *factor*.

        Parameters
        ----------
        factor : int
            Bin factor for this region.
        row_start, row_stop, col_start, col_stop : int
            Rectangle in full-resolution pixel coordinates. Snapped onto the
            block grid via :meth:`region_bounds`.

        Returns
        -------
        arrays : tuple of numpy.ndarray
            ``(mean, real, imag)`` for the region.
        origin : tuple of int
            ``(row_start, col_start)`` after snapping, in full-resolution
            pixels. Use it as the layer ``translate`` (scaled by *factor*)
            so the region sits where it belongs in world coordinates.
        """
        factor = max(1, int(factor))
        row_start, row_stop, col_start, col_stop = self.region_bounds(
            factor, row_start, row_stop, col_start, col_stop
        )
        rows = slice(row_start, row_stop)
        cols = slice(col_start, col_stop)

        mean = self.mean[..., rows, cols]
        real = self.real[..., rows, cols]
        imag = self.imag[..., rows, cols]

        if factor > 1:
            mean, real, imag = bin_phasor_arrays(mean, real, imag, factor)
        return (mean, real, imag), (row_start, col_start)

    def factor_for_region(
        self, row_start, row_stop, col_start, col_stop, budget=None
    ):
        """Return the finest factor keeping a region within *budget* pixels.

        This is what drives zoom refinement: as the viewport shrinks, the
        same pixel budget buys progressively more detail, reaching full
        resolution once the region is small enough to afford it.
        """
        budget = DEFAULT_PIXEL_BUDGET if budget is None else int(budget)
        height = max(1, int(row_stop) - int(row_start))
        width = max(1, int(col_stop) - int(col_start))
        lead = self.full_shape[:-2]
        shape = (*lead, height, width)
        return bin_factor_for_shape(shape, budget)
