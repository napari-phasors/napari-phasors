"""Tests for mosaic (tile) stitching."""

import numpy as np
import pytest
import tifffile
from phasorpy.phasor import phasor_from_signal

from napari_phasors._reader import (
    TileSet,
    raw_file_tile_reader,
    read_tile_phasors,
)
from napari_phasors._stitching import (
    TileGeometry,
    TilePlacement,
    blend_phasor_tiles,
    compute_origins,
    estimate_overlap,
    feather_window,
    layout_from_filenames,
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
