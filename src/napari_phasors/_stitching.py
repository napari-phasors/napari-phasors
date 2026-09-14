"""
This module contains the geometry and blending maths used to stitch tiled
acquisitions (mosaics) into a single phasor image.

Stitching is performed in phasor space rather than on the raw decay or
spectral data.  This is not an approximation: because the discrete Fourier
transform is linear and every tile shares the same number of histogram bins,
summing the raw signals of two overlapping acquisitions is *identical* to
combining their phasor coordinates weighted by mean intensity::

    mean_AB = mean_A + mean_B
    G_AB = (mean_A * G_A + mean_B * G_B) / (mean_A + mean_B)

Working in phasor space means each tile costs ``1 + 2 * n_harmonics`` planes
instead of one plane per histogram bin, which is typically a one to two order
of magnitude reduction in memory for FLIM data.

The module is deliberately free of Qt and napari imports so the maths can be
exercised without a running event loop.
"""

import os
import re
from dataclasses import dataclass, field, replace

import numpy as np

from ._parallel import band_bounds, parallel_map

__all__ = [
    "TilePlacement",
    "TileGeometry",
    "TileSource",
    "as_tile_sources",
    "blend_phasor_tiles",
    "compute_origins",
    "estimate_overlap",
    "feather_ramps",
    "feather_window",
    "layout_from_filenames",
    "layout_from_positions",
    "layout_from_rows",
    "layout_from_stage_positions",
    "parse_tiles_per_row",
]

#: Traversal orders understood by :func:`layout_from_rows`.
TRAVERSAL_ORDERS = ("raster", "snake")

#: Corner the first tile is placed at, for :func:`layout_from_rows`.
START_CORNERS = ("top-left", "top-right", "bottom-left", "bottom-right")

#: Scratch-memory budget for one band of :func:`blend_phasor_tiles`. Only the
#: bands in flight hold accumulators, so peak usage is roughly this times the
#: worker count regardless of how large the mosaic is.
BLEND_BAND_BUDGET_BYTES = 32 << 20

#: Intensity blending modes understood by :func:`blend_phasor_tiles`.
BLEND_MODES = ("feather", "average", "sum")


@dataclass(frozen=True)
class TileSource:
    """Where a single tile comes from.

    A mosaic may be spread over one file per tile, or held entirely in one
    file that stores its tiles along a dedicated dimension (the mosaic axis
    of a CZI, for example). Both are described the same way here: a path
    plus the tile's position along that file's tile axis.

    Parameters
    ----------
    path : str
        Path of the file the tile is read from.
    index : int, optional
        Position of the tile along its file's tile axis. ``0`` for files
        that hold a single tile.
    """

    path: str
    index: int = 0

    @property
    def label(self):
        """Short name identifying this tile, for display and metadata."""
        name = os.path.basename(self.path)
        return name if self.index == 0 else f"{name} [{self.index}]"


def as_tile_sources(items):
    """Normalize *items* into a list of :class:`TileSource`.

    Accepts paths, ``(path, index)`` pairs, and :class:`TileSource` objects,
    so callers can pass whichever is most convenient.

    Parameters
    ----------
    items : iterable
        Tile identifiers.

    Returns
    -------
    list of TileSource
    """
    sources = []
    for item in items:
        if isinstance(item, TileSource):
            sources.append(item)
        elif isinstance(item, (tuple, list)):
            path, index = item
            sources.append(TileSource(str(path), int(index)))
        else:
            sources.append(TileSource(str(item)))
    return sources


@dataclass
class TilePlacement:
    """Position of a single tile within a mosaic.

    Parameters
    ----------
    path : str
        Path of the file this tile was read from.
    row : float
        Row coordinate in units of tile steps. Fractional values are allowed
        so rows with different tile counts can be centered against each other
        (and so hexagonal-like packings can be expressed).
    col : float
        Column coordinate in units of tile steps.
    dy : int, optional
        Per-tile correction along Y in pixels, added on top of the regular
        grid position. Filled in by registration refinement.
    dx : int, optional
        Per-tile correction along X in pixels.
    index : int, optional
        Position of this tile along its file's tile axis, for files that
        hold more than one tile.
    """

    path: str
    row: float
    col: float
    dy: int = 0
    dx: int = 0
    index: int = 0

    @property
    def source(self):
        """Return this placement's :class:`TileSource`."""
        return TileSource(self.path, self.index)


@dataclass
class TileGeometry:
    """Full description of a mosaic layout.

    The geometry separates the regular grid (driven by ``overlap_y`` and
    ``overlap_x``) from per-tile corrections stored on each placement. This
    allows the overlap to be re-tuned after import without discarding
    registration results.

    Parameters
    ----------
    tile_shape : tuple of int
        ``(height, width)`` of a single tile, in pixels.
    placements : list of TilePlacement
        One entry per tile.
    overlap_y : float, optional
        Fractional overlap between vertically adjacent tiles, in ``[0, 0.9]``.
    overlap_x : float, optional
        Fractional overlap between horizontally adjacent tiles.
    blend_mode : str, optional
        One of ``'feather'``, ``'average'`` or ``'sum'``. See
        :func:`blend_phasor_tiles`.
    """

    tile_shape: tuple = (0, 0)
    placements: list = field(default_factory=list)
    overlap_y: float = 0.0
    overlap_x: float = 0.0
    blend_mode: str = "feather"

    @property
    def step_y(self):
        """Vertical distance between neighbouring tile origins, in pixels."""
        return max(1, int(round(self.tile_shape[0] * (1.0 - self.overlap_y))))

    @property
    def step_x(self):
        """Horizontal distance between neighbouring tile origins, in pixels."""
        return max(1, int(round(self.tile_shape[1] * (1.0 - self.overlap_x))))

    @property
    def paths(self):
        """Return the tile paths, in placement order."""
        return [placement.path for placement in self.placements]

    @property
    def sources(self):
        """Return the tiles as :class:`TileSource` objects, in placement order."""
        return [placement.source for placement in self.placements]

    def with_overlap(self, overlap_y, overlap_x):
        """Return a copy of this geometry using a different overlap."""
        return replace(
            self,
            overlap_y=float(overlap_y),
            overlap_x=float(overlap_x),
            placements=list(self.placements),
        )

    def origins(self):
        """Return per-tile ``(y, x)`` pixel origins. See :func:`compute_origins`."""
        return compute_origins(self)

    def canvas_shape(self):
        """Return the ``(height, width)`` of the stitched canvas in pixels."""
        origins = compute_origins(self)
        if not origins:
            return (0, 0)
        height, width = self.tile_shape
        max_y = max(origin[0] for origin in origins)
        max_x = max(origin[1] for origin in origins)
        return (max_y + height, max_x + width)

    def to_dict(self):
        """Return a JSON-serializable description of this geometry."""
        return {
            "tile_shape": list(self.tile_shape),
            "overlap_y": self.overlap_y,
            "overlap_x": self.overlap_x,
            "blend_mode": self.blend_mode,
            "placements": [
                {
                    "path": placement.path,
                    "row": placement.row,
                    "col": placement.col,
                    "dy": placement.dy,
                    "dx": placement.dx,
                    "index": placement.index,
                }
                for placement in self.placements
            ],
        }

    @classmethod
    def from_dict(cls, data):
        """Rebuild a geometry from :meth:`to_dict` output."""
        return cls(
            tile_shape=tuple(data.get("tile_shape", (0, 0))),
            overlap_y=float(data.get("overlap_y", 0.0)),
            overlap_x=float(data.get("overlap_x", 0.0)),
            blend_mode=data.get("blend_mode", "feather"),
            placements=[
                TilePlacement(
                    path=item["path"],
                    row=float(item["row"]),
                    col=float(item["col"]),
                    dy=int(item.get("dy", 0)),
                    dx=int(item.get("dx", 0)),
                    index=int(item.get("index", 0)),
                )
                for item in data.get("placements", [])
            ],
        )


def compute_origins(geometry):
    """Return the pixel origin of every tile in *geometry*.

    Origins are shifted so that the top-left of the bounding box of all tiles
    sits at ``(0, 0)``; the mosaic therefore never has negative coordinates
    even when placements use negative row/column values or negative
    per-tile corrections.

    Parameters
    ----------
    geometry : TileGeometry
        Mosaic layout.

    Returns
    -------
    list of tuple of int
        ``(y, x)`` origin of each tile, in placement order.
    """
    if not geometry.placements:
        return []

    step_y = geometry.step_y
    step_x = geometry.step_x

    raw = [
        (
            int(round(placement.row * step_y)) + int(placement.dy),
            int(round(placement.col * step_x)) + int(placement.dx),
        )
        for placement in geometry.placements
    ]

    min_y = min(origin[0] for origin in raw)
    min_x = min(origin[1] for origin in raw)
    return [(y - min_y, x - min_x) for y, x in raw]


def feather_window(shape, overlap_px, edges=(True, True, True, True)):
    """Return a separable linear ramp used to cross-fade overlapping tiles.

    The window is ``1`` over the interior of the tile and ramps down towards
    ``0`` over ``overlap_px`` pixels at each border that abuts a neighbour.
    Borders on the outside of the mosaic are left flat so the outer edge of
    the stitched image does not fade out.

    Parameters
    ----------
    shape : tuple of int
        ``(height, width)`` of the tile.
    overlap_px : tuple of int
        ``(overlap_y, overlap_x)`` in pixels.
    edges : tuple of bool, optional
        Whether the ``(top, bottom, left, right)`` borders should be ramped.

    Returns
    -------
    numpy.ndarray
        Float32 array of *shape* with values in ``(0, 1]``.
    """
    window_y, window_x = feather_ramps(shape, overlap_px, edges)
    return window_y[:, np.newaxis] * window_x[np.newaxis, :]


def feather_ramps(shape, overlap_px, edges=(True, True, True, True)):
    """Return the separable ``(row, column)`` factors of :func:`feather_window`.

    The window is an outer product, so blending a horizontal band of the
    mosaic only needs the rows of the first factor that the band covers.
    Keeping the two 1-D ramps means a large tile's 2-D window is never
    materialized at all.

    Parameters
    ----------
    shape : tuple of int
        ``(height, width)`` of the tile.
    overlap_px : tuple of int
        ``(overlap_y, overlap_x)`` in pixels.
    edges : tuple of bool, optional
        Whether the ``(top, bottom, left, right)`` borders should be ramped.

    Returns
    -------
    tuple of numpy.ndarray
        ``(window_y, window_x)``, float32, of lengths ``height`` and
        ``width``.
    """
    height, width = shape
    overlap_y, overlap_x = overlap_px
    top, bottom, left, right = edges

    def ramp(size, width_px, ramp_start, ramp_end):
        window = np.ones(size, dtype=np.float32)
        width_px = int(min(max(width_px, 0), size // 2))
        if width_px <= 0:
            return window
        # Values strictly greater than zero so a pixel covered by a single
        # tile always keeps a usable weight.
        taper = (np.arange(width_px, dtype=np.float32) + 1.0) / (
            width_px + 1.0
        )
        if ramp_start:
            window[:width_px] = taper
        if ramp_end:
            window[size - width_px :] = taper[::-1]
        return window

    return (
        ramp(height, overlap_y, top, bottom),
        ramp(width, overlap_x, left, right),
    )


def _tile_edges(geometry, index, origins):
    """Return which ``(top, bottom, left, right)`` borders touch a neighbour."""
    height, width = geometry.tile_shape
    y, x = origins[index]
    top = bottom = left = right = False
    for other_index, (other_y, other_x) in enumerate(origins):
        if other_index == index:
            continue
        overlaps_y = (other_y < y + height) and (other_y + height > y)
        overlaps_x = (other_x < x + width) and (other_x + width > x)
        if overlaps_y and overlaps_x:
            if other_y < y:
                top = True
            if other_y > y:
                bottom = True
            if other_x < x:
                left = True
            if other_x > x:
                right = True
    return (top, bottom, left, right)


def blend_phasor_tiles(
    tiles,
    geometry,
    dtype=np.float32,
    progress=None,
):
    """Blend per-tile phasor coordinates into a single mosaic.

    Tiles are combined with photon-count weighting, which reproduces exactly
    what would be obtained by summing the raw signals before the phasor
    transform (see the module docstring).

    Parameters
    ----------
    tiles : sequence of tuple
        One ``(mean, G, S)`` triple per placement in *geometry*. ``mean`` has
        shape ``(height, width)`` and ``G``/``S`` have shape
        ``(n_harmonics, height, width)``.
    geometry : TileGeometry
        Mosaic layout. ``geometry.blend_mode`` selects how intensity is
        normalized:

        ``'feather'``
            Weighted average using a linear cross-fade over the overlap.
            Produces a seamless, uniformly bright intensity image.
        ``'average'``
            Weighted average with flat per-tile weights.
        ``'sum'``
            Photon counts are summed, so overlap regions are brighter but
            carry the full statistics of every contributing tile.

        ``G`` and ``S`` are identical for all three modes; only the intensity
        normalization differs.
    dtype : numpy.dtype, optional
        Floating point type of the returned arrays. Accumulation is always
        performed in float64 for numerical stability.
    progress : callable, optional
        Called with the index of each tile as it is blended.

    Returns
    -------
    mean : numpy.ndarray
        Stitched mean intensity, shape ``(canvas_y, canvas_x)``. Uncovered
        pixels are ``0``.
    real : numpy.ndarray
        Stitched G coordinates, shape ``(n_harmonics, canvas_y, canvas_x)``.
        Uncovered pixels are ``NaN``.
    imag : numpy.ndarray
        Stitched S coordinates, same shape and conventions as *real*.
    coverage : numpy.ndarray
        Number of tiles contributing to each pixel, shape
        ``(canvas_y, canvas_x)``, dtype ``uint16``.

    Raises
    ------
    ValueError
        If the number of tiles does not match the number of placements, or if
        a tile does not match ``geometry.tile_shape``.
    """
    placements = geometry.placements
    if len(tiles) != len(placements):
        raise ValueError(
            f"Got {len(tiles)} tile(s) for {len(placements)} placement(s)."
        )
    if not tiles:
        raise ValueError("No tiles to blend.")

    blend_mode = geometry.blend_mode
    if blend_mode not in BLEND_MODES:
        raise ValueError(
            f"Unknown blend mode {blend_mode!r}, expected one of {BLEND_MODES}."
        )

    height, width = geometry.tile_shape
    origins = compute_origins(geometry)
    canvas_y, canvas_x = geometry.canvas_shape()

    first_real = np.asarray(tiles[0][1])
    n_harmonics = first_real.shape[0] if first_real.ndim == 3 else 1

    # Validate every tile before touching the canvas, so a bad mosaic fails
    # immediately instead of after allocating gigabytes of accumulators.
    prepared = []
    for index, tile in enumerate(tiles):
        mean, real, imag = (np.asarray(part) for part in tile)
        if mean.shape != (height, width):
            raise ValueError(
                f"Tile {index} has shape {mean.shape}, expected "
                f"{(height, width)}."
            )
        if real.ndim == 2:
            real = real[np.newaxis]
            imag = imag[np.newaxis]
        if real.shape[0] != n_harmonics:
            raise ValueError(
                f"Tile {index} has {real.shape[0]} harmonic(s), expected "
                f"{n_harmonics}."
            )
        prepared.append((mean, real, imag))

    overlap_px = (
        int(round(height * geometry.overlap_y)),
        int(round(width * geometry.overlap_x)),
    )
    feathering = blend_mode == "feather" and bool(
        overlap_px[0] or overlap_px[1]
    )
    if feathering:
        # At most sixteen distinct edge combinations exist, and the ramps are
        # 1-D, so every tile's window costs a couple of hundred bytes.
        ramps = {}
        for index in range(len(prepared)):
            key = _tile_edges(geometry, index, origins)
            if key not in ramps:
                ramps[key] = feather_ramps((height, width), overlap_px, key)
        tile_ramps = [
            ramps[_tile_edges(geometry, index, origins)]
            for index in range(len(prepared))
        ]
    else:
        tile_ramps = [None] * len(prepared)

    mean_out = np.zeros((canvas_y, canvas_x), dtype=dtype)
    real_out = np.full((n_harmonics, canvas_y, canvas_x), np.nan, dtype=dtype)
    imag_out = np.full((n_harmonics, canvas_y, canvas_x), np.nan, dtype=dtype)
    coverage = np.zeros((canvas_y, canvas_x), dtype=np.uint16)

    # The float64 accumulators dominate peak memory, so they are allocated per
    # band rather than per canvas: only the bands actually in flight exist at
    # any moment, which is a handful regardless of how large the mosaic is.
    row_bytes = (2 + 2 * n_harmonics) * max(1, canvas_x) * 8
    max_band = max(1, BLEND_BAND_BUDGET_BYTES // row_bytes)

    def blend_band(row_start, row_stop):
        band_rows = row_stop - row_start
        num_mean = np.zeros((band_rows, canvas_x), dtype=np.float64)
        weight = np.zeros((band_rows, canvas_x), dtype=np.float64)
        num_real = np.zeros(
            (n_harmonics, band_rows, canvas_x), dtype=np.float64
        )
        num_imag = np.zeros(
            (n_harmonics, band_rows, canvas_x), dtype=np.float64
        )

        for index, (mean, real, imag) in enumerate(prepared):
            origin_y, origin_x = origins[index]
            top = max(row_start, origin_y)
            bottom = min(row_stop, origin_y + height)
            if bottom <= top:
                continue

            src = slice(top - origin_y, bottom - origin_y)
            rows = slice(top - row_start, bottom - row_start)
            cols = slice(origin_x, origin_x + width)

            # Only the rows this band owns are upcast, so the float64 traffic
            # over the whole mosaic still adds up to one pass per tile.
            mean_band = np.asarray(mean[src], dtype=np.float64)
            real_band = np.asarray(real[:, src], dtype=np.float64)
            imag_band = np.asarray(imag[:, src], dtype=np.float64)

            ramp = tile_ramps[index]
            if ramp is None:
                alpha = np.ones((bottom - top, width), dtype=np.float64)
            else:
                # Build the outer product in float32 and only then widen, so
                # the weights match :func:`feather_window` bit for bit.
                window_y, window_x = ramp
                alpha = (
                    window_y[src, np.newaxis] * window_x[np.newaxis, :]
                ).astype(np.float64)

            # A pixel only contributes where every phasor coordinate is
            # finite; phasor_from_signal yields NaN wherever the signal has
            # no photons.
            valid = np.isfinite(mean_band)
            valid &= np.all(np.isfinite(real_band), axis=0)
            valid &= np.all(np.isfinite(imag_band), axis=0)

            alpha = np.where(valid, alpha, 0.0)
            # Photon weighting. Negative means (possible after background
            # subtraction) would flip the convex combination, so clip at zero.
            photons = np.where(valid, np.clip(mean_band, 0.0, None), 0.0)
            photon_weight = alpha * photons

            weight[rows, cols] += alpha
            num_mean[rows, cols] += photon_weight
            coverage[row_start:row_stop][rows, cols] += valid.astype(np.uint16)

            for harmonic in range(n_harmonics):
                num_real[harmonic, rows, cols] += photon_weight * np.where(
                    valid, real_band[harmonic], 0.0
                )
                num_imag[harmonic, rows, cols] += photon_weight * np.where(
                    valid, imag_band[harmonic], 0.0
                )

        # Normalize straight into the output, so the full-canvas float64
        # quotient the old implementation built never exists.
        has_photons = num_mean > 0
        safe_photons = np.where(has_photons, num_mean, 1.0)
        real_out[:, row_start:row_stop] = np.where(
            has_photons, num_real / safe_photons, np.nan
        ).astype(dtype)
        imag_out[:, row_start:row_stop] = np.where(
            has_photons, num_imag / safe_photons, np.nan
        ).astype(dtype)

        if blend_mode == "sum":
            mean_out[row_start:row_stop] = num_mean.astype(dtype)
        else:
            covered = weight > 0
            safe_weight = np.where(covered, weight, 1.0)
            mean_out[row_start:row_stop] = np.where(
                covered, num_mean / safe_weight, 0.0
            ).astype(dtype)

    bounds = band_bounds(canvas_y, max_band=max_band)
    if progress is None:
        parallel_map(lambda b: blend_band(*b), bounds)
    else:
        # Bands do not line up with tiles, so report each tile index once, in
        # order, spread over the bands as they complete. Callers only use this
        # to drive a progress bar, and it still advances smoothly.
        reported = 0
        n_tiles = len(prepared)

        def report(position):
            nonlocal reported
            target = round((position + 1) * n_tiles / len(bounds))
            while reported < target:
                progress(reported)
                reported += 1

        parallel_map(lambda b: blend_band(*b), bounds, progress=report)

    return mean_out, real_out, imag_out, coverage


def parse_tiles_per_row(text, n_tiles=None):
    """Parse a ``'5, 7, 9'`` style tiles-per-row specification.

    Also accepts a run-length shorthand such as ``'3x9'`` meaning three
    consecutive rows of nine tiles.

    Parameters
    ----------
    text : str
        The specification to parse.
    n_tiles : int, optional
        If given, the parsed counts must sum to this value.

    Returns
    -------
    list of int
        Number of tiles in each row.

    Raises
    ------
    ValueError
        If the text cannot be parsed, contains non-positive counts, or does
        not account for exactly *n_tiles* tiles.
    """
    counts = []
    for token in re.split(r"[,\s]+", str(text).strip()):
        if not token:
            continue
        match = re.fullmatch(r"(\d+)\s*[x*]\s*(\d+)", token, re.IGNORECASE)
        if match:
            repeats, count = int(match.group(1)), int(match.group(2))
            if repeats <= 0 or count <= 0:
                raise ValueError(f"Invalid tiles-per-row entry {token!r}.")
            counts.extend([count] * repeats)
            continue
        try:
            count = int(token)
        except ValueError as error:
            raise ValueError(
                f"Could not parse tiles-per-row entry {token!r}."
            ) from error
        if count <= 0:
            raise ValueError(f"Invalid tiles-per-row entry {token!r}.")
        counts.append(count)

    if not counts:
        raise ValueError("No tiles-per-row values given.")

    if n_tiles is not None and sum(counts) != n_tiles:
        raise ValueError(
            f"Tiles per row sums to {sum(counts)} but {n_tiles} file(s) "
            "were selected."
        )
    return counts


def layout_from_rows(
    tiles,
    tiles_per_row,
    traversal="raster",
    start_corner="top-left",
    alignment="center",
    tile_shape=(0, 0),
    overlap_y=0.0,
    overlap_x=0.0,
    blend_mode="feather",
):
    """Build a :class:`TileGeometry` by dealing *tiles* into rows.

    Rows may hold different numbers of tiles, which is how partially covered
    mosaics (for example ``5, 7, 9, 9, 7, 5`` for a roughly circular sample)
    are described. Short rows are positioned according to *alignment*, using
    half-step offsets when the row lengths differ in parity.

    Parameters
    ----------
    tiles : sequence
        Tiles in acquisition order, as paths, ``(path, index)`` pairs, or
        :class:`TileSource` objects.
    tiles_per_row : sequence of int or str
        Number of tiles in each row, or a string parsed by
        :func:`parse_tiles_per_row`.
    traversal : {'raster', 'snake'}, optional
        ``'raster'`` restarts every row at the same side; ``'snake'``
        alternates direction, as most stage controllers do.
    start_corner : {'top-left', 'top-right', 'bottom-left', 'bottom-right'}, optional
        Corner where the first tile is placed.
    alignment : {'center', 'left', 'right'}, optional
        How rows shorter than the longest row are positioned.
    tile_shape : tuple of int, optional
        ``(height, width)`` of a single tile.
    overlap_y, overlap_x : float, optional
        Fractional overlap between neighbouring tiles.
    blend_mode : str, optional
        Intensity blending mode stored on the geometry.

    Returns
    -------
    TileGeometry

    Raises
    ------
    ValueError
        If the row specification does not account for exactly ``len(tiles)``
        tiles, or if an option is not recognized.
    """
    sources = as_tile_sources(tiles)
    if isinstance(tiles_per_row, str):
        tiles_per_row = parse_tiles_per_row(tiles_per_row, len(sources))
    else:
        tiles_per_row = [int(count) for count in tiles_per_row]
        if sum(tiles_per_row) != len(sources):
            raise ValueError(
                f"Tiles per row sums to {sum(tiles_per_row)} but "
                f"{len(sources)} tile(s) were selected."
            )

    if traversal not in TRAVERSAL_ORDERS:
        raise ValueError(
            f"Unknown traversal {traversal!r}, expected one of "
            f"{TRAVERSAL_ORDERS}."
        )
    if start_corner not in START_CORNERS:
        raise ValueError(
            f"Unknown start corner {start_corner!r}, expected one of "
            f"{START_CORNERS}."
        )
    if alignment not in ("center", "left", "right"):
        raise ValueError(
            f"Unknown alignment {alignment!r}, expected 'center', 'left' or "
            "'right'."
        )

    longest = max(tiles_per_row)
    flip_x = start_corner.endswith("right")
    flip_y = start_corner.startswith("bottom")

    placements = []
    cursor = 0
    for row_index, count in enumerate(tiles_per_row):
        row_sources = sources[cursor : cursor + count]
        cursor += count

        if alignment == "center":
            offset = (longest - count) / 2.0
        elif alignment == "right":
            offset = float(longest - count)
        else:
            offset = 0.0

        columns = [offset + position for position in range(count)]
        reverse = traversal == "snake" and row_index % 2 == 1
        if reverse != flip_x:
            columns = columns[::-1]

        row = (
            float(len(tiles_per_row) - 1 - row_index)
            if flip_y
            else float(row_index)
        )
        for source, column in zip(row_sources, columns, strict=True):
            placements.append(
                TilePlacement(
                    path=source.path,
                    row=row,
                    col=column,
                    index=source.index,
                )
            )

    return TileGeometry(
        tile_shape=tuple(tile_shape),
        placements=placements,
        overlap_y=float(overlap_y),
        overlap_x=float(overlap_x),
        blend_mode=blend_mode,
    )


def layout_from_filenames(
    tiles,
    pattern=r"(?P<row>\d+)\D+(?P<col>\d+)",
    tile_shape=(0, 0),
    overlap_y=0.0,
    overlap_x=0.0,
    blend_mode="feather",
    **row_kwargs,
):
    """Build a :class:`TileGeometry` from indices encoded in the file names.

    The *pattern* is matched (with :func:`re.search`) against each file's base
    name. It must provide either ``row`` and ``col`` named groups, or a single
    ``index`` group; in the latter case the indices are dealt into rows using
    :func:`layout_from_rows`, so ``tiles_per_row`` must be supplied through
    *row_kwargs*.

    Only applies when every tile lives in its own file: a file name cannot
    distinguish the tiles held inside a single multi-tile file.

    Parameters
    ----------
    tiles : sequence
        Tiles as paths, ``(path, index)`` pairs, or :class:`TileSource`
        objects.
    pattern : str, optional
        Regular expression with ``row``/``col`` or ``index`` named groups.
    tile_shape : tuple of int, optional
        ``(height, width)`` of a single tile.
    overlap_y, overlap_x : float, optional
        Fractional overlap between neighbouring tiles.
    blend_mode : str, optional
        Intensity blending mode stored on the geometry.
    **row_kwargs
        Forwarded to :func:`layout_from_rows` when *pattern* yields a single
        ``index`` group.

    Returns
    -------
    TileGeometry

    Raises
    ------
    ValueError
        If several tiles share a file, if the pattern is invalid, does not
        match every file, or provides neither the ``row``/``col`` pair nor an
        ``index`` group.
    """
    sources = as_tile_sources(tiles)
    paths = [source.path for source in sources]
    if len(set(paths)) != len(paths):
        raise ValueError(
            "File names cannot place tiles that share a file. Use the row "
            "layout instead."
        )
    try:
        regex = re.compile(pattern)
    except re.error as error:
        raise ValueError(f"Invalid file name pattern: {error}") from error

    group_names = set(regex.groupindex)
    has_row_col = {"row", "col"} <= group_names
    has_index = "index" in group_names
    if not (has_row_col or has_index):
        raise ValueError(
            "File name pattern must define 'row' and 'col' groups, or an "
            "'index' group."
        )

    parsed = []
    for path in paths:
        match = regex.search(os.path.basename(path))
        if match is None:
            raise ValueError(
                f"File name pattern did not match {os.path.basename(path)!r}."
            )
        if has_row_col:
            parsed.append((int(match.group("row")), int(match.group("col"))))
        else:
            parsed.append(int(match.group("index")))

    if has_index:
        ordered = [path for _, path in sorted(zip(parsed, paths, strict=True))]
        return layout_from_rows(
            ordered,
            tile_shape=tile_shape,
            overlap_y=overlap_y,
            overlap_x=overlap_x,
            blend_mode=blend_mode,
            **row_kwargs,
        )

    # Indices in file names are arbitrary labels; rank them so that gaps in
    # the numbering do not open gaps in the mosaic.
    row_ranks = {
        value: rank
        for rank, value in enumerate(sorted({r for r, _ in parsed}))
    }
    col_ranks = {
        value: rank
        for rank, value in enumerate(sorted({c for _, c in parsed}))
    }

    placements = [
        TilePlacement(
            path=path,
            row=float(row_ranks[row]),
            col=float(col_ranks[col]),
        )
        for path, (row, col) in zip(paths, parsed, strict=True)
    ]
    return TileGeometry(
        tile_shape=tuple(tile_shape),
        placements=placements,
        overlap_y=float(overlap_y),
        overlap_x=float(overlap_x),
        blend_mode=blend_mode,
    )


def layout_from_positions(
    tiles,
    positions,
    tile_shape,
    blend_mode="feather",
):
    """Build a :class:`TileGeometry` from exact tile positions in pixels.

    Some formats record where every tile sits, so the layout does not have to
    be described or guessed at all. The positions are still expressed in the
    usual grid terms, with a row, a column and an overlap, so the overlap
    stays adjustable afterwards; whatever the regular grid does not account
    for is kept per tile in ``dy``/``dx``, making the reconstruction exact.

    Parameters
    ----------
    tiles : sequence
        Tiles as paths, ``(path, index)`` pairs, or :class:`TileSource`
        objects.
    positions : sequence of tuple
        ``(y, x)`` pixel position of each tile, in the same order as *tiles*.
        Any common offset is removed.
    tile_shape : tuple of int
        ``(height, width)`` of a single tile.
    blend_mode : str, optional
        Intensity blending mode stored on the geometry.

    Returns
    -------
    TileGeometry

    Raises
    ------
    ValueError
        If the number of positions does not match the number of tiles.
    """
    sources = as_tile_sources(tiles)
    positions = [(int(y), int(x)) for y, x in positions]
    if len(positions) != len(sources):
        raise ValueError(
            f"Got {len(positions)} position(s) for {len(sources)} tile(s)."
        )
    if not sources:
        return TileGeometry(tile_shape=tuple(tile_shape))

    height, width = int(tile_shape[0]), int(tile_shape[1])
    ys = np.array([p[0] for p in positions], dtype=float)
    xs = np.array([p[1] for p in positions], dtype=float)
    ys -= ys.min()
    xs -= xs.min()

    # Tiles nominally in the same row or column are rarely at exactly the
    # same coordinate, so group them before ranking.
    row_ranks = _cluster_positions(ys, tolerance=height * 0.5)
    col_ranks = _cluster_positions(xs, tolerance=width * 0.5)

    step_y = _median_step(ys, row_ranks) or height
    step_x = _median_step(xs, col_ranks) or width
    overlap_y = float(np.clip(1.0 - step_y / height, 0.0, 0.9))
    overlap_x = float(np.clip(1.0 - step_x / width, 0.0, 0.9))

    # Rebuild the grid the geometry will actually use, so the residuals below
    # are measured against it rather than against the raw median step.
    grid_step_y = max(1, int(round(height * (1.0 - overlap_y))))
    grid_step_x = max(1, int(round(width * (1.0 - overlap_x))))

    placements = [
        TilePlacement(
            path=source.path,
            row=float(row),
            col=float(col),
            dy=int(round(y)) - int(round(row * grid_step_y)),
            dx=int(round(x)) - int(round(col * grid_step_x)),
            index=source.index,
        )
        for source, row, col, y, x in zip(
            sources, row_ranks, col_ranks, ys, xs, strict=True
        )
    ]

    return TileGeometry(
        tile_shape=(height, width),
        placements=placements,
        overlap_y=overlap_y,
        overlap_x=overlap_x,
        blend_mode=blend_mode,
    )


def _cluster_positions(values, tolerance):
    """Group nearly-equal *values* and return the rank of each value."""
    order = np.argsort(values)
    ranks = np.zeros(len(values), dtype=int)
    current_rank = 0
    for position, index in enumerate(order):
        if position > 0:
            previous = values[order[position - 1]]
            if abs(values[index] - previous) > tolerance:
                current_rank += 1
        ranks[index] = current_rank
    return ranks


def layout_from_stage_positions(
    tiles,
    tile_shape=(0, 0),
    blend_mode="feather",
):
    """Build a :class:`TileGeometry` from stage positions stored in the files.

    Only OME-TIFF files are supported; other formats rarely carry absolute
    stage coordinates. The physical pixel size is used to convert the plane
    positions to pixels, from which both the grid indices and the actual
    overlap fractions are derived.

    Parameters
    ----------
    tiles : sequence
        Tiles as paths, ``(path, index)`` pairs, or :class:`TileSource`
        objects. Tiles sharing a file cannot be placed this way, since an
        OME-TIFF records one position per file.
    tile_shape : tuple of int
        ``(height, width)`` of a single tile, needed to convert the measured
        step into an overlap fraction.
    blend_mode : str, optional
        Intensity blending mode stored on the geometry.

    Returns
    -------
    TileGeometry or None
        ``None`` if positions could not be read for every file, so the caller
        can fall back to another layout source.
    """
    paths = [source.path for source in as_tile_sources(tiles)]
    if len(set(paths)) != len(paths):
        return None
    positions = []
    for path in paths:
        position = _read_stage_position(path)
        if position is None:
            return None
        positions.append(position)

    height, width = tile_shape
    if not height or not width:
        return None

    y_um = np.array([position[0] for position in positions], dtype=float)
    x_um = np.array([position[1] for position in positions], dtype=float)
    pixel_y = np.array([position[2] for position in positions], dtype=float)
    pixel_x = np.array([position[3] for position in positions], dtype=float)

    if not np.all(pixel_y > 0) or not np.all(pixel_x > 0):
        return None

    # Positions in pixels, relative to the top-left tile.
    y_px = (y_um - y_um.min()) / pixel_y.mean()
    x_px = (x_um - x_um.min()) / pixel_x.mean()

    row_ranks = _cluster_positions(y_px, tolerance=height * 0.25)
    col_ranks = _cluster_positions(x_px, tolerance=width * 0.25)

    step_y = _median_step(y_px, row_ranks)
    step_x = _median_step(x_px, col_ranks)
    overlap_y = 0.0 if step_y is None else max(0.0, 1.0 - step_y / height)
    overlap_x = 0.0 if step_x is None else max(0.0, 1.0 - step_x / width)

    placements = [
        TilePlacement(path=path, row=float(row), col=float(col))
        for path, row, col in zip(paths, row_ranks, col_ranks, strict=True)
    ]
    return TileGeometry(
        tile_shape=tuple(tile_shape),
        placements=placements,
        overlap_y=float(min(overlap_y, 0.9)),
        overlap_x=float(min(overlap_x, 0.9)),
        blend_mode=blend_mode,
    )


def _median_step(positions, ranks):
    """Return the median distance between consecutive rank groups."""
    unique_ranks = sorted({int(rank) for rank in ranks})
    if len(unique_ranks) < 2:
        return None
    centers = [
        float(np.mean(positions[ranks == rank])) for rank in unique_ranks
    ]
    steps = np.diff(centers)
    steps = steps[steps > 0]
    if steps.size == 0:
        return None
    return float(np.median(steps))


def _read_stage_position(path):
    """Return ``(y_um, x_um, pixel_y_um, pixel_x_um)`` for an OME-TIFF tile.

    Returns ``None`` when the file is not an OME-TIFF or does not carry the
    required metadata.
    """
    if not str(path).lower().endswith((".ome.tif", ".ome.tiff")):
        return None
    try:
        import tifffile

        with tifffile.TiffFile(path) as tif:
            ome_xml = tif.ome_metadata
            if not ome_xml:
                return None
            ome = tifffile.xml2dict(ome_xml).get("OME", {})
            images = ome.get("Image", [])
            if isinstance(images, dict):
                images = [images]
            if not images:
                return None
            pixels = images[0].get("Pixels", {})

            def _value(container, key):
                value = container.get(f"@{key}", container.get(key))
                if isinstance(value, dict):
                    value = value.get("#text")
                return None if value is None else float(value)

            pixel_x = _value(pixels, "PhysicalSizeX")
            pixel_y = _value(pixels, "PhysicalSizeY")
            if not pixel_x or not pixel_y:
                return None

            planes = pixels.get("Plane", [])
            if isinstance(planes, dict):
                planes = [planes]
            if not planes:
                return None

            position_x = _value(planes[0], "PositionX")
            position_y = _value(planes[0], "PositionY")
            if position_x is None or position_y is None:
                return None
            return (position_y, position_x, pixel_y, pixel_x)
    except Exception:  # noqa: BLE001 - metadata is best effort
        return None


def _overlap_ncc_profile(reference, moving, min_overlap, max_overlap):
    """Score every candidate horizontal overlap between two adjacent tiles.

    For an overlap of ``v`` pixels the right-hand strip ``reference[:, -v:]``
    should reproduce the left-hand strip ``moving[:, :v]``. This function
    returns the normalized cross-correlation of that pair of strips for every
    ``v`` in ``[min_overlap, max_overlap]``.

    All candidates are evaluated in one pass: the cross terms come from an
    FFT cross-correlation along X, and the per-window sums and sums of
    squares from prefix and suffix cumulative sums. Cost is therefore
    ``O(H * W log W)`` for the whole profile rather than per candidate.

    Normalized cross-correlation is used in preference to the raw phase
    correlation peak because microscopy tiles are typically smooth and
    band-limited, and spectral whitening then amplifies frequencies that
    carry no real signal.

    Parameters
    ----------
    reference, moving : numpy.ndarray
        Equally shaped 2D intensity images; *moving* is the right neighbour.
    min_overlap, max_overlap : int
        Inclusive bounds on the overlap width to consider, in pixels.

    Returns
    -------
    overlaps : numpy.ndarray
        The candidate overlap widths that were evaluated.
    scores : numpy.ndarray
        Correlation for each candidate, in ``[-1, 1]``. Degenerate candidates
        (a flat strip on either side) score ``0``.
    """
    height, width = reference.shape
    min_overlap = int(max(1, min_overlap))
    max_overlap = int(min(max_overlap, width))
    if max_overlap < min_overlap:
        return np.empty(0, dtype=int), np.empty(0, dtype=float)

    overlaps = np.arange(min_overlap, max_overlap + 1)

    column_sum_a = reference.sum(axis=0)
    column_sq_a = np.square(reference).sum(axis=0)
    column_sum_b = moving.sum(axis=0)
    column_sq_b = np.square(moving).sum(axis=0)

    # Sums over reference's last v columns and moving's first v columns.
    suffix_sum_a = np.concatenate(([0.0], np.cumsum(column_sum_a[::-1])))
    suffix_sq_a = np.concatenate(([0.0], np.cumsum(column_sq_a[::-1])))
    prefix_sum_b = np.concatenate(([0.0], np.cumsum(column_sum_b)))
    prefix_sq_b = np.concatenate(([0.0], np.cumsum(column_sq_b)))

    # cross[lag] = sum_y sum_j reference[y, lag + j] * moving[y, j], which for
    # lag = width - v is exactly the product term of an overlap of v.
    n_fft = int(2 ** np.ceil(np.log2(max(2 * width, 2))))
    spectrum_a = np.fft.rfft(reference, n=n_fft, axis=1)
    spectrum_b = np.fft.rfft(moving, n=n_fft, axis=1)
    cross_full = np.fft.irfft(
        spectrum_a * np.conj(spectrum_b), n=n_fft, axis=1
    ).sum(axis=0)
    cross = cross_full[width - overlaps]

    counts = height * overlaps
    sum_a = suffix_sum_a[overlaps]
    sum_b = prefix_sum_b[overlaps]
    covariance = cross - sum_a * sum_b / counts
    variance_a = suffix_sq_a[overlaps] - np.square(sum_a) / counts
    variance_b = prefix_sq_b[overlaps] - np.square(sum_b) / counts

    denominator = np.sqrt(
        np.clip(variance_a, 0, None) * np.clip(variance_b, 0, None)
    )
    scores = np.divide(
        covariance,
        denominator,
        out=np.zeros_like(covariance, dtype=float),
        where=denominator > 0,
    )
    return overlaps, np.clip(scores, -1.0, 1.0)


def _best_overlap(reference, moving, min_overlap, max_overlap, axis):
    """Return the best ``(overlap_px, score)`` along *axis* (0=Y, 1=X)."""
    reference = np.nan_to_num(np.asarray(reference, dtype=np.float64))
    moving = np.nan_to_num(np.asarray(moving, dtype=np.float64))
    if reference.shape != moving.shape or min(reference.shape) < 2:
        return None, 0.0
    if axis == 0:
        reference, moving = reference.T, moving.T

    overlaps, scores = _overlap_ncc_profile(
        reference, moving, min_overlap, max_overlap
    )
    if overlaps.size == 0:
        return None, 0.0
    best = int(np.argmax(scores))
    return int(overlaps[best]), float(scores[best])


def estimate_overlap(
    means,
    geometry,
    search_fraction=0.45,
    min_score=0.3,
    min_overlap_px=8,
):
    """Estimate the fractional overlap that best aligns neighbouring tiles.

    Only the mean intensity images are needed, so this runs on the cached
    phasor results without touching the raw files again.

    For every pair of tiles that are neighbours in the layout, each candidate
    overlap is scored by the normalized cross-correlation of the strips it
    would make coincide, and the best scoring width is kept. Pairs whose best
    score falls below *min_score* are discarded, which removes tiles holding
    nothing but background, and the survivors are pooled with a median so a
    few bad pairs cannot skew the result.

    The search covers overlaps only, that is translations along the axis
    joining the two tiles; correcting a perpendicular drift is the job of a
    per-tile registration pass rather than of a single global coefficient.

    Parameters
    ----------
    means : sequence of numpy.ndarray
        Mean intensity image of each tile, in placement order.
    geometry : TileGeometry
        Current layout. Only the neighbour relations and tile shape are used,
        so the estimate does not depend on the current overlap values. If the
        geometry has no tile shape yet, it is taken from *means*.
    search_fraction : float, optional
        Largest overlap considered, as a fraction of the tile size.
    min_score : float, optional
        Minimum normalized cross-correlation for a pair to be trusted.
    min_overlap_px : int, optional
        Smallest overlap considered, in pixels. Very narrow strips correlate
        spuriously well, so this should stay well above a few pixels.

    Returns
    -------
    overlap_y : float or None
        Refined vertical overlap, or ``None`` if no vertical pair could be
        matched confidently.
    overlap_x : float or None
        Refined horizontal overlap, or ``None`` if no horizontal pair could
        be matched confidently.
    """
    placements = geometry.placements
    if len(means) != len(placements) or len(means) < 2:
        return None, None

    # The layout is often built before the tiles have been read, so fall
    # back to the shape of the data rather than refusing to estimate.
    height, width = geometry.tile_shape
    if not height or not width:
        height, width = np.shape(means[0])
    if not height or not width:
        return None, None

    rows = np.array([placement.row for placement in placements])
    cols = np.array([placement.col for placement in placements])

    max_overlap_y = int(round(height * search_fraction))
    max_overlap_x = int(round(width * search_fraction))

    def _to_fraction(measurements, size):
        if not measurements:
            return None
        overlap_px = float(np.median(measurements))
        return float(np.clip(overlap_px / size, 0.0, 0.9))

    # Horizontal pass. Tiles in a row are always a whole step apart, so their
    # strips line up without any cropping.
    horizontal = []
    for index in range(len(placements)):
        for other in range(len(placements)):
            if other == index:
                continue
            if abs(rows[index] - rows[other]) > 1e-6:
                continue
            if abs(cols[other] - cols[index] - 1.0) > 1e-6:
                continue
            overlap, score = _best_overlap(
                means[index],
                means[other],
                min_overlap_px,
                max_overlap_x,
                axis=1,
            )
            if overlap is not None and score >= min_score:
                horizontal.append(overlap)

    overlap_x = _to_fraction(horizontal, width)

    # Vertical pass. Rows of unequal length are centered against each other,
    # so tiles in consecutive rows are generally offset by a fraction of a
    # step and only share part of their columns. Crop to the shared columns
    # first, using the horizontal overlap just measured to convert the
    # column offset into pixels.
    step_x = (
        max(1, int(round(width * (1.0 - overlap_x))))
        if overlap_x is not None
        else geometry.step_x
    )

    vertical = []
    for index in range(len(placements)):
        for other in range(len(placements)):
            if other == index:
                continue
            if abs(rows[other] - rows[index] - 1.0) > 1e-6:
                continue
            column_offset = int(round((cols[other] - cols[index]) * step_x))
            if abs(column_offset) >= width - min_overlap_px:
                continue

            upper, lower = means[index], means[other]
            if column_offset >= 0:
                upper = upper[:, column_offset:]
                lower = lower[:, : width - column_offset]
            else:
                upper = upper[:, : width + column_offset]
                lower = lower[:, -column_offset:]

            overlap, score = _best_overlap(
                upper, lower, min_overlap_px, max_overlap_y, axis=0
            )
            if overlap is not None and score >= min_score:
                vertical.append(overlap)

    return _to_fraction(vertical, height), overlap_x
