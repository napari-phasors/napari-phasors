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

__all__ = [
    "TilePlacement",
    "TileGeometry",
    "blend_phasor_tiles",
    "compute_origins",
    "estimate_overlap",
    "feather_window",
    "layout_from_filenames",
    "layout_from_rows",
    "layout_from_stage_positions",
    "parse_tiles_per_row",
]

#: Traversal orders understood by :func:`layout_from_rows`.
TRAVERSAL_ORDERS = ("raster", "snake")

#: Corner the first tile is placed at, for :func:`layout_from_rows`.
START_CORNERS = ("top-left", "top-right", "bottom-left", "bottom-right")

#: Intensity blending modes understood by :func:`blend_phasor_tiles`.
BLEND_MODES = ("feather", "average", "sum")


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
    """

    path: str
    row: float
    col: float
    dy: int = 0
    dx: int = 0


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

    window_y = ramp(height, overlap_y, top, bottom)
    window_x = ramp(width, overlap_x, left, right)
    return window_y[:, np.newaxis] * window_x[np.newaxis, :]


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

    height, width = geometry.tile_shape
    origins = compute_origins(geometry)
    canvas_y, canvas_x = geometry.canvas_shape()

    first_real = np.asarray(tiles[0][1])
    n_harmonics = first_real.shape[0] if first_real.ndim == 3 else 1

    num_mean = np.zeros((canvas_y, canvas_x), dtype=np.float64)
    weight = np.zeros((canvas_y, canvas_x), dtype=np.float64)
    num_real = np.zeros((n_harmonics, canvas_y, canvas_x), dtype=np.float64)
    num_imag = np.zeros((n_harmonics, canvas_y, canvas_x), dtype=np.float64)
    coverage = np.zeros((canvas_y, canvas_x), dtype=np.uint16)

    overlap_px = (
        int(round(height * geometry.overlap_y)),
        int(round(width * geometry.overlap_x)),
    )
    blend_mode = geometry.blend_mode
    if blend_mode not in BLEND_MODES:
        raise ValueError(
            f"Unknown blend mode {blend_mode!r}, expected one of {BLEND_MODES}."
        )

    for index, (tile, (origin_y, origin_x)) in enumerate(
        zip(tiles, origins, strict=True)
    ):
        mean, real, imag = tile
        mean = np.asarray(mean, dtype=np.float64)
        real = np.asarray(real, dtype=np.float64)
        imag = np.asarray(imag, dtype=np.float64)

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

        rows = slice(origin_y, origin_y + height)
        cols = slice(origin_x, origin_x + width)

        if blend_mode == "feather" and (overlap_px[0] or overlap_px[1]):
            alpha = feather_window(
                (height, width),
                overlap_px,
                edges=_tile_edges(geometry, index, origins),
            ).astype(np.float64)
        else:
            alpha = np.ones((height, width), dtype=np.float64)

        # A pixel only contributes where every phasor coordinate is finite;
        # phasor_from_signal yields NaN wherever the signal has no photons.
        valid = np.isfinite(mean)
        valid &= np.all(np.isfinite(real), axis=0)
        valid &= np.all(np.isfinite(imag), axis=0)

        alpha = np.where(valid, alpha, 0.0)
        # Photon weighting. Negative means (possible after background
        # subtraction) would flip the convex combination, so clip at zero.
        photons = np.where(valid, np.clip(mean, 0.0, None), 0.0)
        photon_weight = alpha * photons

        weight[rows, cols] += alpha
        num_mean[rows, cols] += photon_weight
        coverage[rows, cols] += valid.astype(np.uint16)

        for harmonic in range(n_harmonics):
            num_real[harmonic, rows, cols] += photon_weight * np.where(
                valid, real[harmonic], 0.0
            )
            num_imag[harmonic, rows, cols] += photon_weight * np.where(
                valid, imag[harmonic], 0.0
            )

        if progress is not None:
            progress(index)

    has_photons = num_mean > 0
    safe_photons = np.where(has_photons, num_mean, 1.0)
    real_out = np.where(has_photons, num_real / safe_photons, np.nan)
    imag_out = np.where(has_photons, num_imag / safe_photons, np.nan)

    if blend_mode == "sum":
        mean_out = num_mean
    else:
        safe_weight = np.where(weight > 0, weight, 1.0)
        mean_out = np.where(weight > 0, num_mean / safe_weight, 0.0)

    return (
        mean_out.astype(dtype),
        real_out.astype(dtype),
        imag_out.astype(dtype),
        coverage,
    )


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
    paths,
    tiles_per_row,
    traversal="raster",
    start_corner="top-left",
    alignment="center",
    tile_shape=(0, 0),
    overlap_y=0.0,
    overlap_x=0.0,
    blend_mode="feather",
):
    """Build a :class:`TileGeometry` by dealing *paths* into rows.

    Rows may hold different numbers of tiles, which is how partially covered
    mosaics (for example ``5, 7, 9, 9, 7, 5`` for a roughly circular sample)
    are described. Short rows are positioned according to *alignment*, using
    half-step offsets when the row lengths differ in parity.

    Parameters
    ----------
    paths : sequence of str
        Tile file paths, in acquisition order.
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
        If the row specification does not account for exactly ``len(paths)``
        tiles, or if an option is not recognized.
    """
    paths = list(paths)
    if isinstance(tiles_per_row, str):
        tiles_per_row = parse_tiles_per_row(tiles_per_row, len(paths))
    else:
        tiles_per_row = [int(count) for count in tiles_per_row]
        if sum(tiles_per_row) != len(paths):
            raise ValueError(
                f"Tiles per row sums to {sum(tiles_per_row)} but "
                f"{len(paths)} file(s) were selected."
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
        row_paths = paths[cursor : cursor + count]
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
        for path, column in zip(row_paths, columns, strict=True):
            placements.append(TilePlacement(path=path, row=row, col=column))

    return TileGeometry(
        tile_shape=tuple(tile_shape),
        placements=placements,
        overlap_y=float(overlap_y),
        overlap_x=float(overlap_x),
        blend_mode=blend_mode,
    )


def layout_from_filenames(
    paths,
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

    Parameters
    ----------
    paths : sequence of str
        Tile file paths.
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
        If the pattern is invalid, does not match every file, or provides
        neither the ``row``/``col`` pair nor an ``index`` group.
    """
    paths = list(paths)
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
    paths,
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
    paths : sequence of str
        Tile file paths.
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
    paths = list(paths)
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
