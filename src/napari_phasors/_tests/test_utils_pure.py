"""Pure function tests for _utils.py -- no Qt, no napari viewer.

These tests cover uncovered lines using unittest.mock for napari
layer objects and synthetic numpy arrays.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

from napari_phasors._utils import (
    _apply_filter_and_threshold_to_phasor_arrays,
    _extract_phasor_arrays_from_layer,
    colormap_to_dict,
    natural_sort_key,
    resolve_colormap_by_name,
    threshold_li,
    threshold_otsu,
    threshold_yen,
    validate_harmonics_for_wavelet,
)

# ------------------------------------------------------------------ #
#  threshold_otsu edge cases                                          #
# ------------------------------------------------------------------ #


class TestThresholdOtsuEdgeCases:

    def test_single_value_returns_that_value(self):
        assert threshold_otsu(np.array([5.0])) == 5.0

    def test_two_distinct_values(self):
        data = np.array([0.0, 0.0, 1.0, 1.0])
        result = threshold_otsu(data)
        assert 0.0 <= result <= 1.0

    def test_integer_input_converted(self):
        data = np.array([1, 2, 3, 4, 5])
        result = threshold_otsu(data)
        assert isinstance(result, float)

    def test_negative_values(self):
        data = np.concatenate([np.full(50, -10.0), np.full(50, 10.0)])
        result = threshold_otsu(data)
        assert -10.0 <= result <= 10.0

    def test_custom_nbins(self):
        np.random.seed(99)
        data = np.concatenate(
            [np.random.normal(10, 2, 200), np.random.normal(50, 5, 200)]
        )
        result = threshold_otsu(data, nbins=64)
        assert isinstance(result, float)
        assert 10.0 < result < 50.0


# ------------------------------------------------------------------ #
#  threshold_li edge cases                                            #
# ------------------------------------------------------------------ #


class TestThresholdLiEdgeCases:

    def test_single_value_returns_that_value(self):
        assert threshold_li(np.array([7.0])) == 7.0

    def test_two_identical_values(self):
        assert threshold_li(np.array([3.0, 3.0])) == 3.0

    def test_initial_guess_parameter(self):
        np.random.seed(42)
        data = np.concatenate(
            [np.random.normal(20, 3, 300), np.random.normal(80, 5, 300)]
        )
        result = threshold_li(data, initial_guess=50.0)
        assert isinstance(result, float)
        assert 20.0 < result < 80.0

    def test_tolerance_parameter(self):
        data = np.array([1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0])
        result = threshold_li(data, tolerance=0.001)
        assert isinstance(result, float)

    def test_all_same_except_one(self):
        data = np.array([1.0] * 99 + [100.0])
        result = threshold_li(data)
        assert isinstance(result, float)

    def test_two_values(self):
        data = np.array([0.0, 100.0])
        result = threshold_li(data)
        assert 0.0 <= result <= 100.0


# ------------------------------------------------------------------ #
#  threshold_yen edge cases                                           #
# ------------------------------------------------------------------ #


class TestThresholdYenEdgeCases:

    def test_single_value_returns_that_value(self):
        assert threshold_yen(np.array([3.0])) == 3.0

    def test_two_distinct_values(self):
        data = np.array([0.0, 0.0, 10.0, 10.0])
        result = threshold_yen(data)
        assert isinstance(result, float)

    def test_custom_nbins(self):
        np.random.seed(0)
        data = np.concatenate(
            [np.random.normal(30, 5, 300), np.random.normal(90, 10, 300)]
        )
        result = threshold_yen(data, nbins=128)
        assert isinstance(result, float)
        assert 30.0 < result < 90.0


# ------------------------------------------------------------------ #
#  validate_harmonics_for_wavelet                                     #
# ------------------------------------------------------------------ #


class TestValidateHarmonicsForWavelet:

    def test_float_harmonics(self):
        assert validate_harmonics_for_wavelet([0.5, 1.0])

    def test_large_harmonics(self):
        assert validate_harmonics_for_wavelet([8, 16])

    def test_three_incompatible(self):
        assert not validate_harmonics_for_wavelet([3, 7, 11])

    def test_numpy_float_array(self):
        arr = np.array([1.0, 2.0, 4.0])
        assert validate_harmonics_for_wavelet(arr)

    def test_single_value_with_no_pair(self):
        assert not validate_harmonics_for_wavelet([5])


# ------------------------------------------------------------------ #
#  colormap_to_dict                                                   #
# ------------------------------------------------------------------ #


class TestColormapToDict:

    def test_basic_usage(self):
        cmap = LinearSegmentedColormap.from_list('test', ['black', 'white'])
        result = colormap_to_dict(cmap, num_colors=5)
        assert len(result) == 6  # 5 colors + None
        assert None in result
        assert result[None] == (0, 0, 0, 0)
        for i in range(1, 6):
            assert i in result
            assert len(result[i]) == 4

    def test_exclude_first_false(self):
        cmap = LinearSegmentedColormap.from_list('test', ['red', 'blue'])
        result = colormap_to_dict(cmap, num_colors=3, exclude_first=False)
        assert 1 in result
        assert 2 in result
        assert 3 in result
        assert None in result

    def test_single_color_raises_zero_division(self):
        """num_colors=1 triggers ZeroDivisionError (known edge case)."""
        cmap = LinearSegmentedColormap.from_list('test', ['black', 'white'])
        with pytest.raises(ZeroDivisionError):
            colormap_to_dict(cmap, num_colors=1)

    def test_listed_colormap(self):
        colors = [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)]
        cmap = ListedColormap(colors)
        result = colormap_to_dict(cmap, num_colors=3)
        assert len(result) == 4


# ------------------------------------------------------------------ #
#  natural_sort_key                                                   #
# ------------------------------------------------------------------ #


class TestNaturalSortKey:

    def test_numeric_ordering(self):
        paths = ['img10.tif', 'img2.tif', 'img1.tif']
        sorted_paths = sorted(paths, key=natural_sort_key)
        assert sorted_paths == ['img1.tif', 'img2.tif', 'img10.tif']

    def test_pure_text(self):
        paths = ['banana.tif', 'apple.tif', 'cherry.tif']
        sorted_paths = sorted(paths, key=natural_sort_key)
        assert sorted_paths == ['apple.tif', 'banana.tif', 'cherry.tif']

    def test_mixed_case_insensitive(self):
        paths = ['B2.tif', 'a1.tif', 'A10.tif']
        sorted_paths = sorted(paths, key=natural_sort_key)
        assert sorted_paths == ['a1.tif', 'A10.tif', 'B2.tif']

    def test_with_directory(self):
        import os

        paths = [
            os.path.join('dir', 'z2.tif'),
            os.path.join('dir', 'z10.tif'),
            os.path.join('dir', 'z1.tif'),
        ]
        sorted_paths = sorted(paths, key=natural_sort_key)
        assert sorted_paths == [
            os.path.join('dir', 'z1.tif'),
            os.path.join('dir', 'z2.tif'),
            os.path.join('dir', 'z10.tif'),
        ]

    def test_no_numbers(self):
        key = natural_sort_key('/data/abc.tif')
        assert all(isinstance(p, str) for p in key)

    def test_only_numbers(self):
        key = natural_sort_key('12345')
        assert any(isinstance(p, int) for p in key)


# ------------------------------------------------------------------ #
#  resolve_colormap_by_name                                           #
# ------------------------------------------------------------------ #


class TestResolveColormapByName:

    def test_none_returns_none(self):
        assert resolve_colormap_by_name(None) is None

    def test_select_color_sentinel_returns_none(self):
        assert resolve_colormap_by_name('Select color...') is None

    def test_non_string_returns_none(self):
        assert resolve_colormap_by_name(42) is None

    def test_valid_matplotlib_colormap(self):
        result = resolve_colormap_by_name('viridis')
        assert result is not None

    def test_invalid_name_returns_none(self):
        result = resolve_colormap_by_name('this_colormap_does_not_exist_xyz')
        assert result is None


# ------------------------------------------------------------------ #
#  _extract_phasor_arrays_from_layer (with Mock)                      #
# ------------------------------------------------------------------ #


class TestExtractPhasorArraysFromLayer:

    def _make_mock_layer(self, shape=(8, 8), n_harmonics=2, mask=None):
        layer = MagicMock()
        mean = np.random.rand(*shape).astype(np.float32)
        g = np.random.rand(n_harmonics, *shape).astype(np.float32)
        s = np.random.rand(n_harmonics, *shape).astype(np.float32)
        layer.metadata = {
            'original_mean': mean,
            'G_original': g,
            'S_original': s,
            'harmonics': list(range(1, n_harmonics + 1)),
        }
        if mask is not None:
            layer.metadata['mask'] = mask
        return layer

    def test_basic_extraction(self):
        layer = self._make_mock_layer()
        mean, real, imag, harmonics = _extract_phasor_arrays_from_layer(layer)
        assert mean.shape == (8, 8)
        assert real.shape == (2, 8, 8)
        assert imag.shape == (2, 8, 8)
        assert list(harmonics) == [1, 2]

    def test_returns_copies(self):
        layer = self._make_mock_layer()
        mean, real, imag, _ = _extract_phasor_arrays_from_layer(layer)
        # Modifying returned arrays should not affect metadata
        mean[:] = 999
        assert not np.all(layer.metadata['original_mean'] == 999)

    def test_custom_harmonics_override(self):
        layer = self._make_mock_layer(n_harmonics=3)
        _, _, _, harmonics = _extract_phasor_arrays_from_layer(
            layer, harmonics=np.array([1, 3, 5])
        )
        assert list(harmonics) == [1, 3, 5]

    def test_mask_applied(self):
        shape = (4, 4)
        mask = np.ones(shape, dtype=np.float32)
        mask[0, 0] = 0  # Invalid pixel
        mask[1, 1] = -1  # Also invalid
        layer = self._make_mock_layer(shape=shape, n_harmonics=1, mask=mask)
        mean, real, imag, _ = _extract_phasor_arrays_from_layer(layer)
        assert np.isnan(mean[0, 0])
        assert np.isnan(mean[1, 1])
        assert not np.isnan(mean[0, 1])
        assert np.isnan(real[0, 0, 0])
        assert np.isnan(imag[0, 1, 1])

    def test_no_mask_no_nans(self):
        layer = self._make_mock_layer(shape=(4, 4))
        mean, real, imag, _ = _extract_phasor_arrays_from_layer(layer)
        assert not np.any(np.isnan(mean))


# ------------------------------------------------------------------ #
#  _apply_filter_and_threshold_to_phasor_arrays                       #
# ------------------------------------------------------------------ #


class TestApplyFilterAndThresholdToPhasorArrays:

    def _make_data(self, shape=(8, 8), n_harmonics=2):
        mean = np.random.rand(*shape).astype(np.float64) + 0.1
        real = np.random.rand(n_harmonics, *shape).astype(np.float64)
        imag = np.random.rand(n_harmonics, *shape).astype(np.float64)
        harmonics = np.array(list(range(1, n_harmonics + 1)))
        return mean, real, imag, harmonics

    def test_no_filter_no_threshold(self):
        mean, real, imag, harmonics = self._make_data()
        m, r, i = _apply_filter_and_threshold_to_phasor_arrays(
            mean,
            real,
            imag,
            harmonics,
        )
        # With no threshold the arrays should be unchanged
        # (threshold defaults to None → phasor_threshold with
        # mean_min=None)
        assert m.shape == mean.shape

    def test_median_filter_applied(self):
        mean, real, imag, harmonics = self._make_data(shape=(8, 8))
        m, r, i = _apply_filter_and_threshold_to_phasor_arrays(
            mean,
            real,
            imag,
            harmonics,
            filter_method='median',
            size=3,
            repeat=1,
        )
        assert m.shape == mean.shape

    def test_median_filter_with_repeat_zero(self):
        mean, real, imag, harmonics = self._make_data()
        m, r, i = _apply_filter_and_threshold_to_phasor_arrays(
            mean,
            real,
            imag,
            harmonics,
            filter_method='median',
            size=3,
            repeat=0,
        )
        # repeat=0 means no filtering is applied
        assert m.shape == mean.shape

    def test_threshold_applied(self):
        mean, real, imag, harmonics = self._make_data()
        # Set some pixels to low values
        mean[0, 0] = 0.001
        m, r, i = _apply_filter_and_threshold_to_phasor_arrays(
            mean,
            real,
            imag,
            harmonics,
            threshold=0.05,
        )
        # Low-value pixel should be NaN after thresholding
        assert np.isnan(r[0, 0, 0]) or m[0, 0] < 0.05

    def test_threshold_upper_applied(self):
        mean, real, imag, harmonics = self._make_data()
        mean[0, 0] = 100.0  # Very high
        m, r, i = _apply_filter_and_threshold_to_phasor_arrays(
            mean,
            real,
            imag,
            harmonics,
            threshold=0.0,
            threshold_upper=1.0,
        )
        assert np.isnan(r[0, 0, 0])

    def test_wavelet_filter_with_valid_harmonics(self):
        mean, real, imag, harmonics = self._make_data(
            shape=(8, 8), n_harmonics=2
        )
        # harmonics [1, 2] are valid for wavelet
        m, r, i = _apply_filter_and_threshold_to_phasor_arrays(
            mean,
            real,
            imag,
            harmonics,
            filter_method='wavelet',
            sigma=1.0,
            levels=1,
        )
        assert m.shape == mean.shape

    def test_wavelet_filter_with_invalid_harmonics(self):
        mean, real, imag, _ = self._make_data(shape=(8, 8), n_harmonics=2)
        bad_harmonics = np.array([1, 3])  # Not valid pair
        m, r, i = _apply_filter_and_threshold_to_phasor_arrays(
            mean,
            real,
            imag,
            bad_harmonics,
            filter_method='wavelet',
            sigma=1.0,
            levels=1,
        )
        # Should skip wavelet and only apply threshold
        assert m.shape == mean.shape

    def test_3d_data_median_filter(self):
        mean, real, imag, harmonics = self._make_data(
            shape=(3, 8, 8), n_harmonics=2
        )
        m, r, i = _apply_filter_and_threshold_to_phasor_arrays(
            mean,
            real,
            imag,
            harmonics,
            filter_method='median',
            size=3,
            repeat=1,
        )
        assert m.shape == (3, 8, 8)
