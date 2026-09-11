"""Tests for photon-weighted phasor binning and the level-of-detail cache."""

import numpy as np
import pytest
from phasorpy.phasor import phasor_from_signal

from napari_phasors._binning import (
    PhasorPyramid,
    bin_factor_for_shape,
    bin_phasor_arrays,
    binned_shape,
)


def test_binning_equals_transforming_the_summed_signal():
    """The defining property: binning must not be an approximation.

    Binning phasor coordinates with photon weighting has to give exactly what
    phasor-transforming the summed raw signal would have given. If this drifts,
    a binned level is no longer physically the same measurement.
    """
    rng = np.random.default_rng(3)
    signal = rng.poisson(40, size=(64, 16, 16)).astype(np.float64)
    mean, real, imag = phasor_from_signal(signal, axis=0, harmonic=[1, 2])

    summed = signal.reshape(64, 8, 2, 8, 2).sum(axis=(2, 4))
    _, real_ref, imag_ref = phasor_from_signal(summed, axis=0, harmonic=[1, 2])

    _, real_binned, imag_binned = bin_phasor_arrays(mean, real, imag, 2)

    assert np.allclose(real_binned, real_ref, atol=1e-6, equal_nan=True)
    assert np.allclose(imag_binned, imag_ref, atol=1e-6, equal_nan=True)


def test_binning_is_photon_weighted_not_a_plain_mean():
    """A bright pixel must dominate its block; a plain mean would not."""
    mean = np.array([[100.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    real = np.array([[0.9, 0.1], [0.1, 0.1]], dtype=np.float32)
    imag = np.zeros((2, 2), dtype=np.float32)

    _, binned_real, _ = bin_phasor_arrays(mean, real, imag, 2)

    expected = (100 * 0.9 + 3 * 0.1) / 103
    assert binned_real.shape == (1, 1)
    assert binned_real[0, 0] == pytest.approx(expected, rel=1e-5)
    # The unweighted mean would have been 0.3 -- markedly different.
    assert binned_real[0, 0] > 0.8


def test_intensity_is_averaged_so_levels_share_a_scale():
    """Thresholds set at one level must stay meaningful at another."""
    mean = np.full((4, 4), 7.0, dtype=np.float32)
    real = np.full((4, 4), 0.5, dtype=np.float32)
    imag = np.full((4, 4), 0.2, dtype=np.float32)

    binned_mean, _, _ = bin_phasor_arrays(mean, real, imag, 2)
    assert np.allclose(binned_mean, 7.0)


def test_invalid_pixels_are_excluded_from_the_weighting():
    """NaN coordinates must not poison their block."""
    mean = np.array([[10.0, 10.0], [10.0, 10.0]], dtype=np.float32)
    real = np.array([[0.4, np.nan], [0.4, 0.4]], dtype=np.float32)
    imag = np.zeros((2, 2), dtype=np.float32)

    _, binned_real, _ = bin_phasor_arrays(mean, real, imag, 2)
    assert binned_real[0, 0] == pytest.approx(0.4)


def test_block_with_no_valid_pixel_is_nan():
    mean = np.full((2, 2), np.nan, dtype=np.float32)
    real = np.full((2, 2), np.nan, dtype=np.float32)
    imag = np.full((2, 2), np.nan, dtype=np.float32)

    binned_mean, binned_real, _ = bin_phasor_arrays(mean, real, imag, 2)
    assert np.isnan(binned_mean).all()
    assert np.isnan(binned_real).all()


def test_negative_intensity_carries_no_weight():
    """Background subtraction can go negative; it must not flip the average."""
    mean = np.array([[-50.0, 10.0], [10.0, 10.0]], dtype=np.float32)
    real = np.array([[0.9, 0.2], [0.2, 0.2]], dtype=np.float32)
    imag = np.zeros((2, 2), dtype=np.float32)

    _, binned_real, _ = bin_phasor_arrays(mean, real, imag, 2)
    assert binned_real[0, 0] == pytest.approx(0.2)


def test_partial_blocks_are_padded_not_trimmed():
    """Edge pixels must survive, and bins must stay aligned to the origin."""
    assert binned_shape((7, 5), 2) == (4, 3)
    mean = np.ones((7, 5), dtype=np.float32)
    binned, _, _ = bin_phasor_arrays(mean, mean.copy(), mean.copy(), 2)
    assert binned.shape == (4, 3)
    assert not np.isnan(binned).any()


def test_harmonic_axis_is_preserved():
    mean = np.ones((4, 4), dtype=np.float32)
    real = np.ones((3, 4, 4), dtype=np.float32)
    binned_mean, binned_real, _ = bin_phasor_arrays(mean, real, real.copy(), 2)
    assert binned_mean.shape == (2, 2)
    assert binned_real.shape == (3, 2, 2)


def test_factor_one_is_a_passthrough():
    mean = np.ones((4, 4))
    out = bin_phasor_arrays(mean, mean, mean, 1)
    assert out[0] is mean


def test_mismatched_shapes_raise():
    mean = np.ones((4, 4))
    with pytest.raises(ValueError, match="disagree in shape"):
        bin_phasor_arrays(mean, np.ones((4, 4)), np.ones((2, 2)), 2)
    with pytest.raises(ValueError, match="do not match a mean"):
        bin_phasor_arrays(mean, np.ones((8, 8)), np.ones((8, 8)), 2)


def test_bin_factor_scales_with_image_size():
    assert bin_factor_for_shape((512, 512)) == 1
    assert bin_factor_for_shape((8000, 8000)) > 1
    # Leading axes count towards the budget: a stack bins harder.
    assert bin_factor_for_shape((40, 2048, 2048)) > bin_factor_for_shape(
        (2048, 2048)
    )


def test_bin_factors_are_powers_of_two():
    """Nested levels let a region be refined without rebuilding full res."""
    for shape in [(3000, 3000), (10000, 10000), (60000, 60000)]:
        factor = bin_factor_for_shape(shape)
        assert factor & (factor - 1) == 0


class TestPhasorPyramid:
    """The level cache backing zoom refinement."""

    @staticmethod
    def _pyramid(size=64):
        rng = np.random.default_rng(0)
        mean = rng.random((size, size)).astype(np.float32) * 100
        real = rng.random((2, size, size)).astype(np.float32)
        imag = rng.random((2, size, size)).astype(np.float32)
        return PhasorPyramid(mean, real, imag), mean

    def test_level_one_returns_the_source_arrays(self):
        pyramid, mean = self._pyramid()
        assert pyramid.level(1)[0] is mean

    def test_levels_are_cached(self):
        pyramid, _ = self._pyramid()
        first = pyramid.level(4)
        assert pyramid.level(4)[0] is first[0]
        assert pyramid.available_factors() == [4]
        assert pyramid.nbytes() > 0

    def test_least_recently_used_levels_are_evicted(self):
        pyramid, _ = self._pyramid()
        pyramid.max_cached_levels = 2
        for factor in (2, 4, 8):
            pyramid.level(factor)
        assert pyramid.available_factors() == [4, 8]

    def test_region_matches_the_whole_level(self):
        """A refined region must line up exactly with the coarse grid.

        This is what keeps world coordinates stable when the viewer swaps
        between a whole-image level and a zoomed region.
        """
        pyramid, _ = self._pyramid()
        whole_mean = pyramid.level(4)[0]
        (region_mean, _, _), origin = pyramid.region(4, 16, 48, 8, 40)

        row, col = origin[0] // 4, origin[1] // 4
        expected = whole_mean[
            row : row + region_mean.shape[0],
            col : col + region_mean.shape[1],
        ]
        assert np.allclose(region_mean, expected, equal_nan=True)

    def test_region_start_snaps_onto_the_block_grid(self):
        pyramid, _ = self._pyramid()
        bounds = pyramid.region_bounds(4, 13, 41, 22, 50)
        assert bounds[0] % 4 == 0
        assert bounds[2] % 4 == 0
        assert bounds[0] <= 13 and bounds[2] <= 22

    def test_region_at_factor_one_is_the_raw_slice(self):
        pyramid, mean = self._pyramid()
        (region_mean, _, _), origin = pyramid.region(1, 10, 20, 30, 40)
        assert origin == (10, 30)
        assert np.array_equal(region_mean, mean[10:20, 30:40])

    def test_smaller_regions_earn_finer_factors(self):
        """Zooming in buys detail: the same budget covers fewer pixels."""
        pyramid, _ = self._pyramid()
        whole = pyramid.factor_for_region(0, 64, 0, 64, budget=256)
        part = pyramid.factor_for_region(0, 16, 0, 16, budget=256)
        assert part < whole

    def test_clear_drops_the_cache(self):
        pyramid, _ = self._pyramid()
        pyramid.level(2)
        pyramid.clear()
        assert pyramid.available_factors() == []
        assert pyramid.nbytes() == 0


def test_bin_factor_for_shape_declines_degenerate_input():
    """Nothing to bin: a 1-D shape or a non-positive budget stays at 1."""
    assert bin_factor_for_shape((1024,)) == 1
    assert bin_factor_for_shape(()) == 1
    assert bin_factor_for_shape((8000, 8000), budget=0) == 1
    assert bin_factor_for_shape((8000, 8000), budget=-5) == 1
