"""Tests for mosaic (tile) stitching."""

import numpy as np
import pytest
import tifffile
from phasorpy.phasor import phasor_from_signal

from napari_phasors._reader import (
    TileSet,
    probe_tile_axes,
    raw_file_tile_reader,
    read_tile_phasors,
)
from napari_phasors._stitching import (
    TileGeometry,
    TilePlacement,
    TileSource,
    as_tile_sources,
    blend_phasor_tiles,
    compute_origins,
    estimate_overlap,
    feather_window,
    layout_from_filenames,
    layout_from_positions,
    layout_from_rows,
    layout_from_stage_positions,
    parse_tiles_per_row,
)


def make_scene(shape=(64, 200, 200), seed=0):
    """Return a synthetic FLIM signal with a spatially varying lifetime."""
    import scipy.ndimage as ndi

    rng = np.random.default_rng(seed)
    n_bins, height, width = shape
    lifetime = ndi.gaussian_filter(rng.random((height, width)), 5) * 2 + 1
    time = np.arange(n_bins)[:, None, None]
    return 1000 * np.exp(-time / lifetime[None]) + 5


def write_tiles(directory, scene, geometry, tile_shape):
    """Write one TIFF per placement, cropped out of *scene*. Returns paths."""
    height, width = tile_shape
    step_y, step_x = geometry.step_y, geometry.step_x
    paths = []
    for index, placement in enumerate(geometry.placements):
        origin_y = int(round(placement.row * step_y))
        origin_x = int(round(placement.col * step_x))
        path = directory / f"tile_{index:03d}.tif"
        tifffile.imwrite(
            path,
            scene[
                :,
                origin_y : origin_y + height,
                origin_x : origin_x + width,
            ].astype(np.uint16),
        )
        paths.append(str(path))
    return paths


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------


def test_parse_tiles_per_row():
    assert parse_tiles_per_row("5, 7, 9") == [5, 7, 9]
    assert parse_tiles_per_row("3x9") == [9, 9, 9]
    assert parse_tiles_per_row("2x3, 4") == [3, 3, 4]
    assert parse_tiles_per_row("5 7 9", n_tiles=21) == [5, 7, 9]


@pytest.mark.parametrize("text", ["", "0", "-1", "abc", "2,x", "0x3"])
def test_parse_tiles_per_row_invalid(text):
    with pytest.raises(ValueError):
        parse_tiles_per_row(text)


def test_parse_tiles_per_row_wrong_total():
    with pytest.raises(ValueError, match="sums to"):
        parse_tiles_per_row("3,3", n_tiles=9)


def test_layout_from_rows_ragged_is_centered():
    """A partially covered mosaic keeps short rows centered on the long one."""
    paths = [f"t{i}" for i in range(21)]
    geometry = layout_from_rows(paths, "5,7,9", tile_shape=(10, 10))

    columns = {
        row: sorted(p.col for p in geometry.placements if p.row == row)
        for row in (0, 1, 2)
    }
    assert columns[0] == [2.0, 3.0, 4.0, 5.0, 6.0]
    assert columns[1] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    assert columns[2] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]


def test_layout_from_rows_half_step_offset():
    """Rows whose lengths differ in parity are offset by half a step."""
    geometry = layout_from_rows([f"t{i}" for i in range(11)], "5,6")
    assert [p.col for p in geometry.placements if p.row == 0] == [
        0.5,
        1.5,
        2.5,
        3.5,
        4.5,
    ]
    assert [p.col for p in geometry.placements if p.row == 1] == [
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
    ]


def test_layout_from_rows_snake_reverses_alternate_rows():
    raster = layout_from_rows([f"t{i}" for i in range(6)], "3,3")
    snake = layout_from_rows(
        [f"t{i}" for i in range(6)], "3,3", traversal="snake"
    )
    assert [p.col for p in raster.placements] == [0, 1, 2, 0, 1, 2]
    assert [p.col for p in snake.placements] == [0, 1, 2, 2, 1, 0]


def test_layout_from_rows_start_corner():
    geometry = layout_from_rows(
        ["a", "b", "c", "d"], "2,2", start_corner="bottom-right"
    )
    placed = {p.path: (p.row, p.col) for p in geometry.placements}
    assert placed == {
        "a": (1.0, 1.0),
        "b": (1.0, 0.0),
        "c": (0.0, 1.0),
        "d": (0.0, 0.0),
    }


def test_layout_from_rows_alignment():
    left = layout_from_rows(
        [f"t{i}" for i in range(5)], "2,3", alignment="left"
    )
    right = layout_from_rows(
        [f"t{i}" for i in range(5)], "2,3", alignment="right"
    )
    assert [p.col for p in left.placements if p.row == 0] == [0.0, 1.0]
    assert [p.col for p in right.placements if p.row == 0] == [1.0, 2.0]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"traversal": "spiral"},
        {"start_corner": "middle"},
        {"alignment": "justified"},
    ],
)
def test_layout_from_rows_invalid_options(kwargs):
    with pytest.raises(ValueError):
        layout_from_rows(["a", "b"], "2", **kwargs)


def test_layout_from_rows_count_mismatch():
    with pytest.raises(ValueError, match="sums to"):
        layout_from_rows(["a", "b", "c"], [2, 2])


def test_compute_origins_and_canvas():
    geometry = layout_from_rows(
        [f"t{i}" for i in range(4)],
        "2,2",
        tile_shape=(48, 48),
        overlap_y=0.25,
        overlap_x=0.25,
    )
    assert geometry.step_y == 36 and geometry.step_x == 36
    assert compute_origins(geometry) == [(0, 0), (0, 36), (36, 0), (36, 36)]
    assert geometry.canvas_shape() == (84, 84)


def test_compute_origins_shifts_negative_placements_into_frame():
    geometry = TileGeometry(
        tile_shape=(10, 10),
        placements=[
            TilePlacement("a", -1, -1),
            TilePlacement("b", 0, 0),
        ],
    )
    assert compute_origins(geometry) == [(0, 0), (10, 10)]


def test_geometry_with_overlap_rescales_grid():
    geometry = layout_from_rows(
        [f"t{i}" for i in range(4)], "2,2", tile_shape=(40, 40)
    )
    wider = geometry.with_overlap(0.5, 0.5)
    assert geometry.canvas_shape() == (80, 80)
    assert wider.canvas_shape() == (60, 60)
    # The layout itself is untouched; only the spacing changes.
    assert [p.col for p in wider.placements] == [
        p.col for p in geometry.placements
    ]


def test_geometry_round_trips_through_dict():
    geometry = layout_from_rows(
        [f"t{i}" for i in range(6)],
        "3,3",
        tile_shape=(16, 16),
        overlap_y=0.2,
        overlap_x=0.3,
        blend_mode="sum",
    )
    restored = TileGeometry.from_dict(geometry.to_dict())
    assert restored.tile_shape == geometry.tile_shape
    assert restored.overlap_x == geometry.overlap_x
    assert restored.blend_mode == "sum"
    assert restored.origins() == geometry.origins()


def test_feather_window_ramps_only_requested_edges():
    window = feather_window((10, 10), (3, 3), edges=(False, True, False, True))
    assert np.allclose(window[0], window[1])  # top not ramped
    assert window[-1, 0] < window[5, 0]  # bottom ramped
    assert window[0, -1] < window[0, 5]  # right ramped
    assert np.all(window > 0)


# --------------------------------------------------------------------------
# Blending
# --------------------------------------------------------------------------


def test_blending_matches_summing_the_raw_signals():
    """Blending phasors with photon weights == transforming summed signals.

    This is the property the whole design rests on: because the DFT is
    linear, stitching in phasor space is exact rather than an approximation.
    """
    rng = np.random.default_rng(3)
    n_bins, height, width, overlap = 64, 40, 40, 10
    scene = rng.poisson(30, (n_bins, height, width + width - overlap))
    scene = scene.astype(float)

    left_raw = scene[:, :, :width]
    right_raw = scene[:, :, width - overlap : 2 * width - overlap]
    left = phasor_from_signal(left_raw, axis=0, harmonic=[1, 2])
    right = phasor_from_signal(right_raw, axis=0, harmonic=[1, 2])

    geometry = TileGeometry(
        tile_shape=(height, width),
        overlap_x=overlap / width,
        blend_mode="sum",
        placements=[TilePlacement("a", 0, 0), TilePlacement("b", 0, 1)],
    )
    mean, real, imag, coverage = blend_phasor_tiles(
        [left, right], geometry, dtype=np.float64
    )

    expected = phasor_from_signal(
        left_raw[:, :, width - overlap :] + right_raw[:, :, :overlap],
        axis=0,
        harmonic=[1, 2],
    )
    seam = slice(width - overlap, width)
    assert np.allclose(mean[:, seam], expected[0])
    assert np.allclose(real[:, :, seam], expected[1])
    assert np.allclose(imag[:, :, seam], expected[2])
    assert set(np.unique(coverage)) == {1, 2}


def test_blending_is_seamless_for_uniform_tiles():
    """Feathering a uniform scene must not leave visible seams."""
    height = width = 32
    geometry = layout_from_rows(
        [f"t{i}" for i in range(5)],
        "2,3",
        tile_shape=(height, width),
        overlap_y=0.25,
        overlap_x=0.25,
        blend_mode="feather",
    )
    tiles = [
        (
            np.full((height, width), 100.0),
            np.full((2, height, width), 0.5),
            np.full((2, height, width), 0.3),
        )
        for _ in geometry.placements
    ]
    mean, real, imag, coverage = blend_phasor_tiles(tiles, geometry)

    covered = coverage > 0
    assert np.allclose(mean[covered], 100.0)
    assert np.allclose(real[:, covered], 0.5)
    assert np.allclose(imag[:, covered], 0.3)


def test_uncovered_canvas_is_empty_and_nan():
    """Gaps in a ragged mosaic read as no data, not as zero-lifetime pixels."""
    height = width = 16
    geometry = layout_from_rows(
        ["a", "b", "c"], "1,2", tile_shape=(height, width)
    )
    tiles = [
        (
            np.full((height, width), 10.0),
            np.zeros((1, height, width)),
            np.zeros((1, height, width)),
        )
        for _ in range(3)
    ]
    mean, real, imag, coverage = blend_phasor_tiles(tiles, geometry)

    empty = coverage == 0
    assert empty.any()
    assert np.all(mean[empty] == 0)
    assert np.all(np.isnan(real[:, empty]))
    assert np.all(np.isnan(imag[:, empty]))
    assert not np.isnan(real[:, ~empty]).any()


def test_blend_modes_differ_in_intensity_only():
    """Only the intensity normalization changes; G and S are invariant."""
    height = width = 24
    rng = np.random.default_rng(1)
    placements = [TilePlacement("a", 0, 0), TilePlacement("b", 0, 1)]
    tiles = [
        (
            rng.random((height, width)) * 100 + 1,
            rng.random((1, height, width)),
            rng.random((1, height, width)),
        )
        for _ in placements
    ]

    def blend(mode):
        return blend_phasor_tiles(
            tiles,
            TileGeometry(
                tile_shape=(height, width),
                placements=placements,
                overlap_x=0.25,
                blend_mode=mode,
            ),
            dtype=np.float64,
        )

    average_mean, average_real, _, coverage = blend("average")
    sum_mean, sum_real, _, _ = blend("sum")

    assert np.allclose(average_real, sum_real, equal_nan=True)
    seam = coverage == 2
    assert np.all(sum_mean[seam] > average_mean[seam])
    assert np.allclose(sum_mean[coverage == 1], average_mean[coverage == 1])


def test_all_nan_tile_is_ignored():
    height = width = 12
    placements = [TilePlacement("a", 0, 0), TilePlacement("b", 0, 1)]
    good = (
        np.full((height, width), 5.0),
        np.full((1, height, width), 0.4),
        np.full((1, height, width), 0.2),
    )
    bad = (
        np.full((height, width), np.nan),
        np.full((1, height, width), np.nan),
        np.full((1, height, width), np.nan),
    )
    geometry = TileGeometry(
        tile_shape=(height, width), placements=placements, overlap_x=0.25
    )
    mean, real, imag, coverage = blend_phasor_tiles([good, bad], geometry)

    assert coverage.max() == 1
    assert np.allclose(real[:, coverage > 0], 0.4)


def test_blend_accepts_two_dimensional_phasor_arrays():
    height = width = 8
    tiles = [
        (
            np.ones((height, width)),
            np.full((height, width), 0.6),
            np.full((height, width), 0.1),
        )
    ]
    geometry = TileGeometry(
        tile_shape=(height, width), placements=[TilePlacement("a", 0, 0)]
    )
    _, real, _, _ = blend_phasor_tiles(tiles, geometry)
    assert real.shape == (1, height, width)


def test_blend_rejects_mismatched_input():
    geometry = TileGeometry(
        tile_shape=(8, 8), placements=[TilePlacement("a", 0, 0)]
    )
    with pytest.raises(ValueError, match="placement"):
        blend_phasor_tiles([], geometry)

    wrong = (np.ones((4, 4)), np.ones((1, 4, 4)), np.ones((1, 4, 4)))
    with pytest.raises(ValueError, match="expected"):
        blend_phasor_tiles([wrong], geometry)

    with pytest.raises(ValueError, match="blend mode"):
        blend_phasor_tiles(
            [(np.ones((8, 8)), np.ones((1, 8, 8)), np.ones((1, 8, 8)))],
            TileGeometry(
                tile_shape=(8, 8),
                placements=[TilePlacement("a", 0, 0)],
                blend_mode="nonsense",
            ),
        )


# --------------------------------------------------------------------------
# Overlap estimation
# --------------------------------------------------------------------------


def build_mosaic_means(spec, overlap_y, overlap_x, noise=0.0, **kwargs):
    """Return ``(geometry, means)`` cropped out of a common scene."""
    import scipy.ndimage as ndi

    rng = np.random.default_rng(0)
    height = width = 64
    scene = ndi.gaussian_filter(rng.random((900, 900)), 2) * 1000

    n_tiles = sum(parse_tiles_per_row(spec))
    geometry = layout_from_rows(
        [f"t{i}" for i in range(n_tiles)],
        spec,
        tile_shape=(height, width),
        overlap_y=0.1,
        overlap_x=0.1,
        **kwargs,
    )
    step_y = int(round(height * (1 - overlap_y)))
    step_x = int(round(width * (1 - overlap_x)))
    means = []
    for placement in geometry.placements:
        origin_y = int(round(placement.row * step_y))
        origin_x = int(round(placement.col * step_x))
        tile = scene[origin_y : origin_y + height, origin_x : origin_x + width]
        means.append(tile + rng.normal(0, noise, (height, width)))
    return geometry, means


@pytest.mark.parametrize(
    "overlap_y, overlap_x", [(0.25, 0.25), (0.30, 0.15), (0.20, 0.20)]
)
def test_estimate_overlap_recovers_a_regular_grid(overlap_y, overlap_x):
    geometry, means = build_mosaic_means("3,3,3", overlap_y, overlap_x)
    found_y, found_x = estimate_overlap(means, geometry)
    assert found_y == pytest.approx(overlap_y, abs=0.02)
    assert found_x == pytest.approx(overlap_x, abs=0.02)


def test_estimate_overlap_handles_ragged_rows():
    """Centered rows sit half a step apart, so shared columns must be cropped."""
    geometry, means = build_mosaic_means("5,7,9", 0.25, 0.15)
    found_y, found_x = estimate_overlap(means, geometry)
    assert found_y == pytest.approx(0.25, abs=0.02)
    assert found_x == pytest.approx(0.15, abs=0.02)


def test_estimate_overlap_survives_uninformative_tiles():
    geometry, means = build_mosaic_means("3,3,3", 0.25, 0.25)
    means[4] = np.zeros_like(means[4])
    means[0] = np.full_like(means[0], 100.0)
    found_y, found_x = estimate_overlap(means, geometry)
    assert found_y == pytest.approx(0.25, abs=0.02)
    assert found_x == pytest.approx(0.25, abs=0.02)


def test_estimate_overlap_is_independent_of_the_starting_guess():
    geometry, means = build_mosaic_means("3,3,3", 0.25, 0.25)
    from_low = estimate_overlap(means, geometry.with_overlap(0.01, 0.01))
    from_high = estimate_overlap(means, geometry.with_overlap(0.40, 0.40))
    assert from_low == from_high


def test_estimate_overlap_declines_rather_than_guessing():
    """Tiles that cannot be matched return None instead of a bad number."""
    geometry, _ = build_mosaic_means("3,3,3", 0.25, 0.25)
    rng = np.random.default_rng(5)
    unrelated = [rng.random((64, 64)) for _ in geometry.placements]
    assert estimate_overlap(unrelated, geometry) == (None, None)


def test_estimate_overlap_works_without_a_tile_shape():
    geometry, means = build_mosaic_means("3,3,3", 0.25, 0.25)
    from dataclasses import replace

    shapeless = replace(geometry, tile_shape=(0, 0))
    assert estimate_overlap(means, shapeless)[1] == pytest.approx(
        0.25, abs=0.02
    )


# --------------------------------------------------------------------------
# Layout sources
# --------------------------------------------------------------------------


def test_layout_from_filenames_row_col():
    paths = [
        "s_R01_C03.tif",
        "s_R01_C05.tif",
        "s_R02_C03.tif",
        "s_R02_C05.tif",
    ]
    geometry = layout_from_filenames(paths, r"R(?P<row>\d+)_C(?P<col>\d+)")
    # Non-contiguous indices are ranked so gaps in the numbering do not open
    # gaps in the mosaic.
    assert [(p.row, p.col) for p in geometry.placements] == [
        (0.0, 0.0),
        (0.0, 1.0),
        (1.0, 0.0),
        (1.0, 1.0),
    ]


def test_layout_from_filenames_index_is_sorted_then_dealt():
    paths = ["t_003.tif", "t_001.tif", "t_002.tif", "t_004.tif"]
    geometry = layout_from_filenames(
        paths, r"_(?P<index>\d+)", tiles_per_row="2,2", traversal="snake"
    )
    assert [p.path for p in geometry.placements] == [
        "t_001.tif",
        "t_002.tif",
        "t_003.tif",
        "t_004.tif",
    ]
    assert [p.col for p in geometry.placements] == [0.0, 1.0, 1.0, 0.0]


def test_layout_from_filenames_errors():
    with pytest.raises(ValueError, match="must define"):
        layout_from_filenames(["a.tif"], r"(?P<foo>\d+)")
    with pytest.raises(ValueError, match="did not match"):
        layout_from_filenames(["a.tif"], r"R(?P<row>\d+)_C(?P<col>\d+)")
    with pytest.raises(ValueError, match="Invalid file name pattern"):
        layout_from_filenames(["a.tif"], r"(?P<row>\d+")


def test_layout_from_stage_positions_reads_ome_tiff(tmp_path):
    """Positions and pixel size give both the grid and the true overlap."""
    tile = np.ones((8, 8), dtype=np.uint16)
    paths = []
    for index, (y_um, x_um) in enumerate(
        [(0.0, 0.0), (0.0, 6.0), (6.0, 0.0), (6.0, 6.0)]
    ):
        path = tmp_path / f"tile{index}.ome.tif"
        tifffile.imwrite(
            path,
            tile,
            metadata={
                "PhysicalSizeX": 1.0,
                "PhysicalSizeY": 1.0,
                "Plane": {"PositionX": [x_um], "PositionY": [y_um]},
            },
        )
        paths.append(str(path))

    geometry = layout_from_stage_positions(paths, tile_shape=(8, 8))
    assert geometry is not None
    assert [(p.row, p.col) for p in geometry.placements] == [
        (0.0, 0.0),
        (0.0, 1.0),
        (1.0, 0.0),
        (1.0, 1.0),
    ]
    # A 6 um step for an 8 px tile at 1 um/px is a 25 % overlap.
    assert geometry.overlap_x == pytest.approx(0.25)
    assert geometry.overlap_y == pytest.approx(0.25)


def test_layout_from_stage_positions_returns_none_without_metadata(tmp_path):
    path = tmp_path / "plain.tif"
    tifffile.imwrite(path, np.zeros((4, 4), dtype=np.uint16))
    assert layout_from_stage_positions([str(path)], tile_shape=(4, 4)) is None


# --------------------------------------------------------------------------
# Reader integration
# --------------------------------------------------------------------------


@pytest.fixture
def tile_mosaic(tmp_path):
    """Write a 3x3 mosaic of FLIM tiles with a known 25 % overlap."""
    height = width = 48
    geometry = layout_from_rows(
        [""] * 9,
        "3,3,3",
        tile_shape=(height, width),
        overlap_y=0.25,
        overlap_x=0.25,
    )
    scene = make_scene(shape=(32, 200, 200))
    paths = write_tiles(tmp_path, scene, geometry, (height, width))
    geometry = layout_from_rows(
        paths,
        "3,3,3",
        tile_shape=(height, width),
        overlap_y=0.25,
        overlap_x=0.25,
    )
    return paths, geometry


def test_read_tile_phasors_caches_every_tile(tile_mosaic):
    paths, _ = tile_mosaic
    tile_set = read_tile_phasors(paths, harmonics=[1, 2])

    assert isinstance(tile_set, TileSet)
    assert tile_set.n_tiles == 9
    assert tile_set.n_channels == 1
    assert tile_set.tile_shape == (48, 48)
    assert len(tile_set.means(0)) == 9
    mean, real, imag = tile_set.tiles[0][0]
    assert mean.shape == (48, 48)
    assert real.shape == imag.shape == (2, 48, 48)
    assert tile_set.nbytes() > 0


def test_stitched_layer_matches_the_plugin_metadata_contract(tile_mosaic):
    paths, geometry = tile_mosaic
    layers = read_tile_phasors(paths, harmonics=[1, 2]).stitch(geometry)

    assert len(layers) == 1
    data, add_kwargs = layers[0]
    assert data.shape == (120, 120)
    assert "Mosaic Intensity Image" in add_kwargs["name"]

    metadata = add_kwargs["metadata"]
    for key in (
        "original_mean",
        "settings",
        "G",
        "S",
        "G_original",
        "S_original",
        "harmonics",
    ):
        assert key in metadata
    assert metadata["G"].shape == (2, 120, 120)
    assert metadata["harmonics"] == [1, 2]
    assert np.shares_memory(metadata["G"], metadata["G_original"]) is False
    assert metadata["tile_geometry"]["overlap_x"] == pytest.approx(0.25)
    assert len(metadata["tile_files"]) == 9
    assert set(np.unique(metadata["tile_coverage"])) == {1, 2, 4}


def test_restitching_reuses_the_cache(tile_mosaic):
    """Changing the overlap must not need the files again."""
    paths, geometry = tile_mosaic
    tile_set = read_tile_phasors(paths, harmonics=[1])

    wide = tile_set.stitch(geometry)[0][0]
    narrow = tile_set.stitch(geometry.with_overlap(0.10, 0.10))[0][0]
    assert wide.shape == (120, 120)
    assert narrow.shape == (134, 134)


def test_stitched_mosaic_reproduces_the_source_scene(tile_mosaic):
    """Interior pixels of the mosaic match a single-shot phasor of the scene."""
    paths, geometry = tile_mosaic
    layers = read_tile_phasors(paths, harmonics=[1]).stitch(geometry)
    metadata = layers[0][1]["metadata"]

    scene = make_scene(shape=(32, 200, 200))
    _, real, _ = phasor_from_signal(
        scene[:, :120, :120].astype(np.uint16), axis=0, harmonic=[1]
    )
    # Tolerance covers the uint16 rounding applied when the tiles were saved.
    assert np.allclose(metadata["G"][0], real, atol=1e-3)


def test_estimate_overlap_on_real_tiles(tile_mosaic):
    paths, geometry = tile_mosaic
    tile_set = read_tile_phasors(paths, harmonics=[1])
    found_y, found_x = estimate_overlap(tile_set.means(0), geometry)
    assert found_y == pytest.approx(0.25, abs=0.02)
    assert found_x == pytest.approx(0.25, abs=0.02)


def test_raw_file_tile_reader_end_to_end(tile_mosaic):
    paths, geometry = tile_mosaic
    layers = raw_file_tile_reader(paths, geometry, harmonics=[1])
    assert len(layers) == 1
    assert layers[0][0].shape == (120, 120)


def test_read_tile_phasors_rejects_bad_input(tmp_path, tile_mosaic):
    paths, _ = tile_mosaic

    with pytest.raises(ValueError, match="No files"):
        read_tile_phasors([])

    odd = tmp_path / "other.ome.tif"
    tifffile.imwrite(odd, np.zeros((4, 4), dtype=np.uint16))
    with pytest.raises(ValueError, match="same extension"):
        read_tile_phasors([paths[0], str(odd)])

    mismatched = tmp_path / "small.tif"
    tifffile.imwrite(mismatched, np.zeros((32, 8, 8), dtype=np.uint16))
    with pytest.raises(ValueError, match="Shape mismatch"):
        read_tile_phasors([paths[0], str(mismatched)], harmonics=[1])


def test_stitching_already_processed_tiles(tmp_path):
    """Mosaics of phasor OME-TIFFs stitch as well as mosaics of raw files."""
    from napari_phasors._tests.test_data_utils import get_test_file_path

    source = get_test_file_path("test_file.ome.tif")
    paths = [source] * 4
    tile_set = read_tile_phasors(paths, harmonics=[1])
    assert tile_set.n_tiles == 4

    geometry = layout_from_rows(
        paths,
        "2,2",
        tile_shape=tile_set.tile_shape,
        overlap_y=0.25,
        overlap_x=0.25,
    )
    data, add_kwargs = tile_set.stitch(geometry)[0]

    height, width = tile_set.tile_shape
    assert data.shape == geometry.canvas_shape()
    assert data.shape[0] < 2 * height and data.shape[1] < 2 * width
    # Calibration and other persisted settings survive stitching.
    assert add_kwargs["metadata"]["settings"]


# --------------------------------------------------------------------------
# Mosaics held inside a single file
# --------------------------------------------------------------------------


@pytest.fixture
def single_file_mosaic(tmp_path):
    """Write one TIFF holding a 3x3 mosaic along a leading tile axis.

    The stored array is ``(tile, histogram, Y, X)``, so reading it needs both
    a tile axis of 0 and a phasor axis of 1.
    """
    height = width = 48
    geometry = layout_from_rows(
        [""] * 9,
        "3,3,3",
        tile_shape=(height, width),
        overlap_y=0.25,
        overlap_x=0.25,
    )
    scene = make_scene(shape=(32, 200, 200))
    stack = np.stack(
        [
            scene[
                :,
                int(round(p.row * geometry.step_y)) : int(
                    round(p.row * geometry.step_y)
                )
                + height,
                int(round(p.col * geometry.step_x)) : int(
                    round(p.col * geometry.step_x)
                )
                + width,
            ]
            for p in geometry.placements
        ]
    ).astype(np.uint16)

    path = str(tmp_path / "mosaic.tif")
    tifffile.imwrite(path, stack)

    sources = [TileSource(path, index) for index in range(9)]
    geometry = layout_from_rows(
        sources,
        "3,3,3",
        tile_shape=(height, width),
        overlap_y=0.25,
        overlap_x=0.25,
    )
    return path, sources, geometry


def test_tile_source_normalization():
    assert as_tile_sources(["a.tif"]) == [TileSource("a.tif", 0)]
    assert as_tile_sources([("a.czi", 3)]) == [TileSource("a.czi", 3)]
    assert as_tile_sources([TileSource("a.czi", 1)]) == [
        TileSource("a.czi", 1)
    ]


def test_tile_source_label_distinguishes_tiles_in_one_file():
    assert TileSource("/data/a.czi").label == "a.czi"
    assert TileSource("/data/a.czi", 4).label == "a.czi [4]"


def test_layout_carries_the_tile_index():
    sources = [TileSource("m.czi", index) for index in range(4)]
    geometry = layout_from_rows(sources, "2,2", tile_shape=(8, 8))
    assert [p.index for p in geometry.placements] == [0, 1, 2, 3]
    assert geometry.sources == sources
    assert geometry.paths == ["m.czi"] * 4
    # The index has to survive serialization or a reloaded mosaic would
    # collapse onto a single tile.
    assert TileGeometry.from_dict(geometry.to_dict()).sources == sources


def test_filename_layout_refuses_tiles_sharing_a_file():
    sources = [TileSource("m_1_2.czi", index) for index in range(2)]
    with pytest.raises(ValueError, match="share a file"):
        layout_from_filenames(sources)


def test_stage_position_layout_declines_tiles_sharing_a_file():
    sources = [TileSource("m.ome.tif", index) for index in range(2)]
    assert layout_from_stage_positions(sources, tile_shape=(8, 8)) is None


def test_probe_tile_axes_finds_the_tile_axis(single_file_mosaic):
    path, _, _ = single_file_mosaic
    assert probe_tile_axes(path) == {0: 9}


def test_probe_tile_axes_empty_for_single_tile_files(tile_mosaic):
    paths, _ = tile_mosaic
    assert probe_tile_axes(paths[0]) == {}


def test_probe_tile_axes_survives_unreadable_files(tmp_path):
    path = tmp_path / "broken.tif"
    path.write_bytes(b"not a tiff")
    assert probe_tile_axes(str(path)) == {}


def test_reading_a_mosaic_held_in_one_file(single_file_mosaic):
    path, sources, geometry = single_file_mosaic
    tile_set = read_tile_phasors(
        sources,
        reader_options={"phasor_axis": 1},
        harmonics=[1, 2],
        tile_axis=0,
    )

    assert tile_set.n_tiles == 9
    assert tile_set.n_files == 1
    assert tile_set.tile_shape == (48, 48)
    assert tile_set.n_channels == 1

    data, add_kwargs = tile_set.stitch(geometry)[0]
    assert data.shape == (120, 120)
    assert add_kwargs["metadata"]["G"].shape == (2, 120, 120)
    # The layer is named after the file, not its folder.
    assert add_kwargs["name"].startswith("mosaic Mosaic Intensity Image")
    labels = add_kwargs["metadata"]["tile_files"]
    assert labels[0] == "mosaic.tif"
    assert labels[3] == "mosaic.tif [3]"


def test_one_file_mosaic_matches_the_same_tiles_as_files(
    single_file_mosaic, tmp_path
):
    """Splitting one file gives the same mosaic as one file per tile."""
    path, sources, geometry = single_file_mosaic
    combined = read_tile_phasors(
        sources, reader_options={"phasor_axis": 1}, harmonics=[1], tile_axis=0
    ).stitch(geometry)[0]

    stack = tifffile.imread(path)
    separate_paths = []
    for index in range(stack.shape[0]):
        tile_path = tmp_path / f"separate_{index:03d}.tif"
        tifffile.imwrite(tile_path, stack[index])
        separate_paths.append(str(tile_path))
    separate_geometry = layout_from_rows(
        separate_paths,
        "3,3,3",
        tile_shape=(48, 48),
        overlap_y=0.25,
        overlap_x=0.25,
    )
    separate = read_tile_phasors(separate_paths, harmonics=[1]).stitch(
        separate_geometry
    )[0]

    np.testing.assert_allclose(combined[0], separate[0])
    np.testing.assert_allclose(
        combined[1]["metadata"]["G"],
        separate[1]["metadata"]["G"],
        equal_nan=True,
    )


def test_each_file_is_read_once_however_tiles_are_ordered(
    single_file_mosaic, monkeypatch
):
    path, sources, geometry = single_file_mosaic

    import napari_phasors._reader as reader_module

    reads = []
    original = reader_module.load_raw_signal

    def counting_load(file_path, io_options=None):
        reads.append(file_path)
        return original(file_path, io_options)

    monkeypatch.setattr(reader_module, "load_raw_signal", counting_load)

    shuffled = [sources[i] for i in (4, 0, 8, 2, 6, 1, 7, 3, 5)]
    tile_set = read_tile_phasors(
        shuffled, reader_options={"phasor_axis": 1}, harmonics=[1], tile_axis=0
    )

    assert reads.count(path) == 1
    # Placement order is preserved even though the file was read in index
    # order, so tile 4 is still first.
    assert tile_set.sources == shuffled


def test_tiles_may_repeat_within_a_mosaic(single_file_mosaic):
    path, _, _ = single_file_mosaic
    repeated = [TileSource(path, 0), TileSource(path, 0)]
    tile_set = read_tile_phasors(
        repeated,
        reader_options={"phasor_axis": 1},
        harmonics=[1],
        tile_axis=0,
    )
    assert tile_set.n_tiles == 2
    np.testing.assert_array_equal(tile_set.means(0)[0], tile_set.means(0)[1])


def test_tile_axis_errors_are_actionable(single_file_mosaic):
    path, sources, _ = single_file_mosaic

    with pytest.raises(ValueError, match="no dimension named 'M'"):
        read_tile_phasors(sources, harmonics=[1], tile_axis="M")

    with pytest.raises(ValueError, match="out of range"):
        read_tile_phasors(sources, harmonics=[1], tile_axis=9)

    with pytest.raises(ValueError, match="cannot read tile"):
        read_tile_phasors(
            [TileSource(path, 99)],
            reader_options={"phasor_axis": 1},
            harmonics=[1],
            tile_axis=0,
        )


def test_processed_files_cannot_be_split_into_tiles():
    from napari_phasors._tests.test_data_utils import get_test_file_path

    source = get_test_file_path("test_file.ome.tif")
    with pytest.raises(ValueError, match="cannot be split into tiles"):
        read_tile_phasors(
            [TileSource(source, 0), TileSource(source, 1)], harmonics=[1]
        )


@pytest.mark.parametrize(
    "dims, shape, expected",
    [
        (("M", "H", "Y", "X"), (9, 32, 16, 16), 0),
        (("S", "C", "Y", "X"), (4, 28, 16, 16), 0),
        (("V", "M", "H", "Y", "X"), (2, 9, 32, 16, 16), 1),
    ],
)
def test_tile_axis_detection_prefers_mosaic_dimensions(dims, shape, expected):
    """A CZI mosaic axis is picked over other non-spatial dimensions."""
    import xarray as xr

    from napari_phasors._reader import _resolve_tile_axis

    signal = xr.DataArray(np.zeros(shape), dims=dims)
    assert _resolve_tile_axis(signal, None, "f.czi") == expected


def test_tile_axis_detection_ignores_channel_and_spatial_axes():
    import xarray as xr

    from napari_phasors._reader import _resolve_tile_axis

    signal = xr.DataArray(np.zeros((28, 16, 16)), dims=("C", "Y", "X"))
    with pytest.raises(ValueError, match="no dimension holding tiles"):
        _resolve_tile_axis(signal, None, "f.czi")


def test_phasor_axis_is_corrected_after_the_tile_axis_is_dropped(
    single_file_mosaic,
):
    """Slicing removes an axis, so a later phasor axis must shift down."""
    path, sources, geometry = single_file_mosaic
    tile_set = read_tile_phasors(
        sources[:1],
        reader_options={"phasor_axis": 1},
        harmonics=[1],
        tile_axis=0,
    )
    mean, real, _ = tile_set.tiles[0][0]
    assert mean.shape == (48, 48)
    # A phasor axis left at 1 would have transformed along Y and produced
    # nonsense outside the universal semicircle.
    assert np.all(real[np.isfinite(real)] <= 1.0)
    assert np.all(real[np.isfinite(real)] >= 0.0)


# --------------------------------------------------------------------------
# Mosaics positioned by the file itself (CZI)
# --------------------------------------------------------------------------


def test_layout_from_positions_recovers_a_ragged_grid():
    """Measured positions give back rows, columns and the overlap."""
    height = width = 2048
    step = 1945  # ~5 % overlap, as a slide scanner records it
    rows = [16, 16, 15]
    positions, sources = [], []
    for row, count in enumerate(rows):
        for col in range(count):
            positions.append((94928 + row * step, 169175 + col * step))
            sources.append(TileSource("m.czi", len(sources)))

    geometry = layout_from_positions(sources, positions, (height, width))

    assert geometry.overlap_y == pytest.approx(1 - step / height, abs=1e-3)
    assert geometry.overlap_x == pytest.approx(1 - step / width, abs=1e-3)
    per_row = {}
    for placement in geometry.placements:
        per_row.setdefault(placement.row, []).append(placement.col)
    assert [len(v) for v in per_row.values()] == rows
    # The common offset of the recorded coordinates is removed.
    assert geometry.origins()[0] == (0, 0)
    assert geometry.canvas_shape() == (2 * step + height, 15 * step + width)


def test_layout_from_positions_is_exact():
    """Whatever the regular grid misses is kept per tile, so nothing drifts."""
    height = width = 100
    # Deliberately irregular: a jittery stage.
    positions = [(0, 0), (0, 95), (0, 191), (88, 3), (88, 97), (88, 190)]
    sources = [TileSource("m.czi", i) for i in range(6)]

    geometry = layout_from_positions(sources, positions, (height, width))
    assert geometry.origins() == positions


def test_layout_from_positions_rejects_a_count_mismatch():
    with pytest.raises(ValueError, match="position"):
        layout_from_positions(
            [TileSource("m.czi", 0)], [(0, 0), (0, 90)], (10, 10)
        )


def test_binning_sums_photons():
    """Binning must sum, so a binned tile keeps the photon statistics."""
    from napari_phasors._reader import _bin_spatial

    cube = np.arange(2 * 4 * 4, dtype=np.uint16).reshape(2, 4, 4)
    binned = _bin_spatial(cube, 2)

    assert binned.shape == (2, 2, 2)
    assert binned[0, 0, 0] == cube[0, :2, :2].sum()
    assert binned.sum() == cube.sum()
    assert np.array_equal(_bin_spatial(cube, 1), cube)


def test_binning_preserves_the_phasor_of_the_pooled_pixels():
    """A binned tile's phasor equals the photon-weighted phasor of its pixels.

    This is the same identity stitching relies on, so binning to fit a large
    mosaic in memory does not bias the result.
    """
    from napari_phasors._reader import _bin_spatial

    rng = np.random.default_rng(0)
    cube = rng.poisson(40, (32, 4, 4)).astype(np.uint16)

    mean, real, imag = phasor_from_signal(cube, axis=0, harmonic=[1])
    binned = _bin_spatial(cube, 4)
    binned_mean, binned_real, binned_imag = phasor_from_signal(
        binned, axis=0, harmonic=[1]
    )

    weights = mean / mean.sum()
    assert binned_real[0, 0, 0] == pytest.approx((real[0] * weights).sum())
    assert binned_imag[0, 0, 0] == pytest.approx((imag[0] * weights).sum())


def test_binning_rejects_a_factor_larger_than_the_tile():
    from napari_phasors._reader import _bin_spatial

    with pytest.raises(ValueError, match="leaves nothing"):
        _bin_spatial(np.zeros((2, 4, 4), dtype=np.uint16), 8)


def test_probe_reports_a_czi_mosaic(monkeypatch):
    import napari_phasors._reader as reader_module

    monkeypatch.setattr(
        reader_module,
        "czi_mosaic_info",
        lambda path: {
            "n_tiles": 171,
            "tile_shape": (2048, 2048),
            "canvas_shape": (21504, 31232),
            "n_channels": 32,
        },
    )
    assert reader_module.probe_tile_axes("scan.czi") == {"mosaic": 171}
    assert reader_module.describe_tile_axis("mosaic") == "Mosaic tiles"


def test_a_mosaic_is_never_read_as_one_image(monkeypatch):
    """Reading a mosaic whole would allocate far more than the file holds."""
    import napari_phasors._reader as reader_module

    monkeypatch.setattr(
        reader_module,
        "czi_mosaic_info",
        lambda path: {
            "n_tiles": 171,
            "tile_shape": (2048, 2048),
            "canvas_shape": (21504, 31232),
            "n_channels": 32,
        },
    )
    with pytest.raises(ValueError, match="Open tiled mosaic"):
        reader_module.load_raw_signal("scan.czi")


def test_czi_mosaic_info_is_none_for_other_formats(tile_mosaic):
    from napari_phasors._reader import czi_mosaic_info

    paths, _ = tile_mosaic
    assert czi_mosaic_info(paths[0]) is None


def test_czi_mosaic_info_is_none_for_a_plain_czi():
    from napari_phasors._reader import czi_mosaic_info
    from napari_phasors._tests.test_data_utils import get_test_file_path

    assert czi_mosaic_info(get_test_file_path("test_file.czi")) is None
