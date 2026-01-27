import numpy as np
from matplotlib.colors import ListedColormap
from phasorpy.filter import (
    phasor_filter_median,
    phasor_filter_pawflim,
    phasor_threshold,
)

from napari_phasors._synthetic_generator import (
    make_intensity_layer_with_phasors,
    make_raw_flim_data,
)
from napari_phasors._utils import (
    apply_filter_and_threshold,
    colormap_to_dict,
    update_frequency_in_metadata,
    validate_harmonics_for_wavelet,
)


def test_validate_harmonics_for_wavelet():
    """Test validate_harmonics_for_wavelet function."""
    # Test compatible harmonics (each has double or half)
    compatible_harmonics_1 = [1, 2]  # 2 = 1*2
    assert validate_harmonics_for_wavelet(compatible_harmonics_1) == True

    compatible_harmonics_2 = [2, 4]  # 4 = 2*2
    assert validate_harmonics_for_wavelet(compatible_harmonics_2) == True

    compatible_harmonics_3 = [1, 2, 4]  # 2 = 1*2, 4 = 2*2
    assert validate_harmonics_for_wavelet(compatible_harmonics_3) == True

    compatible_harmonics_4 = [2, 1, 4]  # Same as above, different order
    assert validate_harmonics_for_wavelet(compatible_harmonics_4) == True

    compatible_harmonics_5 = [0.5, 1, 2]  # 1 = 0.5*2, 2 = 1*2
    assert validate_harmonics_for_wavelet(compatible_harmonics_5) == True

    # Test incompatible harmonics (some don't have double or half)
    incompatible_harmonics_1 = [
        1,
        3,
    ]  # 3 has no double/half relationship with 1
    assert validate_harmonics_for_wavelet(incompatible_harmonics_1) == False

    incompatible_harmonics_2 = [1, 3, 5]  # None have double/half relationships
    assert validate_harmonics_for_wavelet(incompatible_harmonics_2) == False

    incompatible_harmonics_3 = [1, 2, 5]  # 1,2 are compatible but 5 is not
    assert validate_harmonics_for_wavelet(incompatible_harmonics_3) == False

    # Test single harmonic (incompatible by definition)
    single_harmonic = [1]
    assert validate_harmonics_for_wavelet(single_harmonic) == False

    # Test empty array
    empty_harmonics = []
    assert (
        validate_harmonics_for_wavelet(empty_harmonics) == True
    )  # Vacuously true

    # Test numpy array input
    numpy_harmonics = np.array([1, 2, 4])
    assert validate_harmonics_for_wavelet(numpy_harmonics) == True

    # Test with duplicates
    harmonics_with_duplicates = [1, 1, 2, 2]
    assert validate_harmonics_for_wavelet(harmonics_with_duplicates) == True


def test_apply_filter_and_threshold_median(make_napari_viewer):
    """Test apply_filter_and_threshold function with median filter."""
    raw_flim_data = make_raw_flim_data(shape=(5, 5))
    harmonic = [1, 2]
    intensity_image_layer = make_intensity_layer_with_phasors(
        raw_flim_data, harmonic=harmonic
    )
    original_mean = intensity_image_layer.metadata['original_mean']
    original_g = intensity_image_layer.metadata['G'].copy()
    original_s = intensity_image_layer.metadata['S'].copy()
    assert np.all(
        intensity_image_layer.data
        == intensity_image_layer.metadata['original_mean']
    )
    assert not np.any(np.isnan(intensity_image_layer.data))
    viewer = make_napari_viewer()
    viewer.add_layer(intensity_image_layer)
    threshold = 0.02

    # Test median filter
    apply_filter_and_threshold(
        intensity_image_layer,
        threshold=threshold,
        filter_method="median",
        size=3,
        repeat=1,
    )

    assert np.all(
        intensity_image_layer.metadata['original_mean'] == original_mean
    )
    assert intensity_image_layer.data.shape == (5, 5)
    assert np.isnan(intensity_image_layer.data[0][0])
    assert not np.isnan(intensity_image_layer.data[0][4])

    # Check original values are preserved
    assert np.all(intensity_image_layer.metadata['G_original'] == original_g)
    assert np.all(intensity_image_layer.metadata['S_original'] == original_s)

    harmonics = intensity_image_layer.metadata['harmonics']
    # Calculate expected filtered values
    filtered_mean, expected_g, expected_s = phasor_filter_median(
        original_mean, original_g, original_s, repeat=1, size=3
    )
    _, expected_g, expected_s = phasor_threshold(
        filtered_mean, expected_g, expected_s, threshold
    )

    filtered_g = intensity_image_layer.metadata['G']
    filtered_s = intensity_image_layer.metadata['S']
    assert np.allclose(expected_g, filtered_g, equal_nan=True)
    assert np.allclose(expected_s, filtered_s, equal_nan=True)

    # Check that settings are saved in metadata
    assert "settings" in intensity_image_layer.metadata
    assert (
        intensity_image_layer.metadata["settings"]["filter"]["method"]
        == "median"
    )
    assert intensity_image_layer.metadata["settings"]["filter"]["size"] == 3
    assert intensity_image_layer.metadata["settings"]["filter"]["repeat"] == 1
    assert intensity_image_layer.metadata["settings"]["threshold"] == threshold


def test_apply_filter_and_threshold_wavelet_compatible(make_napari_viewer):
    """Test apply_filter_and_threshold function with compatible wavelet harmonics."""
    raw_flim_data = make_raw_flim_data(shape=(5, 5))
    harmonic = [1, 2]  # Compatible harmonics
    intensity_image_layer = make_intensity_layer_with_phasors(
        raw_flim_data, harmonic=harmonic
    )
    original_mean = intensity_image_layer.metadata['original_mean'].copy()
    original_g = intensity_image_layer.metadata['G_original'].copy()
    original_s = intensity_image_layer.metadata['S_original'].copy()

    viewer = make_napari_viewer()
    viewer.add_layer(intensity_image_layer)

    threshold = 0.02
    sigma = 2.0
    levels = 1

    # Test wavelet filter with compatible harmonics
    apply_filter_and_threshold(
        intensity_image_layer,
        threshold=threshold,
        filter_method="wavelet",
        sigma=sigma,
        levels=levels,
    )

    # Check that original data is preserved
    assert np.all(
        intensity_image_layer.metadata['original_mean'] == original_mean
    )

    # Check that original phasor values are preserved
    assert np.all(intensity_image_layer.metadata['G_original'] == original_g)
    assert np.all(intensity_image_layer.metadata['S_original'] == original_s)

    # The filtered G and S should exist and be valid arrays
    assert 'G' in intensity_image_layer.metadata
    assert 'S' in intensity_image_layer.metadata
    assert intensity_image_layer.metadata['G'].shape == original_g.shape
    assert intensity_image_layer.metadata['S'].shape == original_s.shape

    # Check that settings are saved
    assert "settings" in intensity_image_layer.metadata
    assert (
        intensity_image_layer.metadata["settings"]["filter"]["method"]
        == "wavelet"
    )
    assert (
        intensity_image_layer.metadata["settings"]["filter"]["sigma"] == sigma
    )
    assert (
        intensity_image_layer.metadata["settings"]["filter"]["levels"]
        == levels
    )
    assert intensity_image_layer.metadata["settings"]["threshold"] == threshold


def test_apply_filter_and_threshold_wavelet_incompatible(make_napari_viewer):
    """Test apply_filter_and_threshold function with incompatible wavelet harmonics."""
    raw_flim_data = make_raw_flim_data(shape=(5, 5))
    # Create layer with incompatible harmonics
    intensity_image_layer = make_intensity_layer_with_phasors(
        raw_flim_data, harmonic=[1, 3]  # Incompatible harmonics
    )
    original_mean = intensity_image_layer.metadata['original_mean'].copy()
    original_g = intensity_image_layer.metadata['G_original'].copy()
    original_s = intensity_image_layer.metadata['S_original'].copy()

    viewer = make_napari_viewer()
    viewer.add_layer(intensity_image_layer)

    threshold = 0.02

    # Test wavelet filter with incompatible harmonics - should only apply threshold
    apply_filter_and_threshold(
        intensity_image_layer,
        threshold=threshold,
        filter_method="wavelet",
        sigma=2.0,
        levels=1,
    )

    # Check that original data is preserved
    assert np.all(
        intensity_image_layer.metadata['original_mean'] == original_mean
    )

    # Since harmonics are incompatible, no wavelet filtering should occur
    # Only thresholding should be applied

    # Apply only threshold (no filtering) for comparison
    _, expected_g, expected_s = phasor_threshold(
        original_mean, original_g, original_s, threshold
    )

    actual_g = intensity_image_layer.metadata['G']
    actual_s = intensity_image_layer.metadata['S']

    assert np.allclose(expected_g, actual_g, equal_nan=True)
    assert np.allclose(expected_s, actual_s, equal_nan=True)


def test_apply_filter_and_threshold_custom_harmonics_values():
    """Test apply_filter_and_threshold with custom harmonics and validate results."""
    raw_flim_data = make_raw_flim_data(shape=(5, 5))
    harmonic = [1, 2]  # Layer harmonics
    intensity_image_layer = make_intensity_layer_with_phasors(
        raw_flim_data, harmonic=harmonic
    )

    original_mean = intensity_image_layer.metadata['original_mean'].copy()
    original_g = intensity_image_layer.metadata['G_original'].copy()
    original_s = intensity_image_layer.metadata['S_original'].copy()

    # Test with custom harmonics that match the layer's harmonics
    custom_harmonics = np.array([1, 2])
    threshold = 0.01
    sigma = 1.5
    levels = 2

    apply_filter_and_threshold(
        intensity_image_layer,
        threshold=threshold,
        filter_method="wavelet",
        sigma=sigma,
        levels=levels,
        harmonics=custom_harmonics,
    )

    # Calculate expected results
    expected_mean, expected_g, expected_s = phasor_filter_pawflim(
        original_mean,
        original_g,
        original_s,
        sigma=sigma,
        levels=levels,
        harmonic=custom_harmonics,
    )
    # Apply threshold to get the final expected values (including mean modifications)
    expected_mean, expected_g, expected_s = phasor_threshold(
        expected_mean, expected_g, expected_s, threshold
    )

    # Compare results
    actual_g = intensity_image_layer.metadata['G']
    actual_s = intensity_image_layer.metadata['S']

    assert np.allclose(expected_g, actual_g, equal_nan=True)
    assert np.allclose(expected_s, actual_s, equal_nan=True)
    assert np.allclose(
        expected_mean, intensity_image_layer.data, equal_nan=True
    )

    # Check metadata
    assert "settings" in intensity_image_layer.metadata
    assert (
        intensity_image_layer.metadata["settings"]["filter"]["method"]
        == "wavelet"
    )


def test_apply_filter_and_threshold_no_filter():
    """Test apply_filter_and_threshold with no filtering (repeat=0 for median)."""
    raw_flim_data = make_raw_flim_data(shape=(5, 5))
    harmonic = [1, 2]
    intensity_image_layer = make_intensity_layer_with_phasors(
        raw_flim_data, harmonic=harmonic
    )
    original_mean = intensity_image_layer.metadata['original_mean'].copy()
    original_g = intensity_image_layer.metadata['G_original'].copy()
    original_s = intensity_image_layer.metadata['S_original'].copy()

    threshold = 0.02

    # Test median filter with repeat=0 (no filtering)
    apply_filter_and_threshold(
        intensity_image_layer,
        threshold=threshold,
        filter_method="median",
        size=3,
        repeat=0,  # No filtering
    )

    # Only thresholding should be applied
    # Apply only threshold for comparison
    _, expected_g, expected_s = phasor_threshold(
        original_mean, original_g, original_s, threshold
    )

    actual_g = intensity_image_layer.metadata['G']
    actual_s = intensity_image_layer.metadata['S']

    assert np.allclose(expected_g, actual_g, equal_nan=True)
    assert np.allclose(expected_s, actual_s, equal_nan=True)


def test_apply_filter_and_threshold_custom_harmonics():
    """Test apply_filter_and_threshold with custom harmonics parameter."""
    raw_flim_data = make_raw_flim_data(shape=(5, 5))
    harmonic = [1, 2]  # Use only the harmonics we'll pass as custom
    intensity_image_layer = make_intensity_layer_with_phasors(
        raw_flim_data, harmonic=harmonic
    )

    # Test with custom harmonics that match the layer's harmonics
    custom_harmonics = np.array([1, 2])

    apply_filter_and_threshold(
        intensity_image_layer,
        threshold=0.01,
        filter_method="wavelet",
        sigma=1.0,
        levels=1,
        harmonics=custom_harmonics,
    )

    # Should work without error
    assert "settings" in intensity_image_layer.metadata
    assert (
        intensity_image_layer.metadata["settings"]["filter"]["method"]
        == "wavelet"
    )


def test_apply_filter_and_threshold_custom_harmonics_subset():
    """Test apply_filter_and_threshold with custom harmonics that are a subset."""
    raw_flim_data = make_raw_flim_data(shape=(5, 5))
    harmonic = [1, 2]  # Layer has these harmonics
    intensity_image_layer = make_intensity_layer_with_phasors(
        raw_flim_data, harmonic=harmonic
    )

    # Get original arrays
    original_g = intensity_image_layer.metadata['G_original'].copy()
    original_s = intensity_image_layer.metadata['S_original'].copy()

    # Update layer to only have harmonic 1 (subset)
    # Take only the first harmonic's data
    intensity_image_layer.metadata['G_original'] = original_g[0:1]
    intensity_image_layer.metadata['S_original'] = original_s[0:1]
    intensity_image_layer.metadata['G'] = original_g[0:1].copy()
    intensity_image_layer.metadata['S'] = original_s[0:1].copy()
    intensity_image_layer.metadata['harmonics'] = [1]

    # Now test with custom harmonics
    custom_harmonics = np.array([1])

    apply_filter_and_threshold(
        intensity_image_layer,
        threshold=0.01,
        filter_method="wavelet",
        sigma=1.0,
        levels=1,
        harmonics=custom_harmonics,
    )

    # Should work without error since harmonics match the modified data
    assert "settings" in intensity_image_layer.metadata
    assert (
        intensity_image_layer.metadata["settings"]["filter"]["method"]
        == "wavelet"
    )


def test_colormap_to_dict():
    """Test colormap_to_dict function."""
    # Create a simple colormap for testing
    colors = [(0, 0, 0, 1), (0.5, 0.5, 0.5, 1), (1, 1, 1, 1)]
    colormap = ListedColormap(colors)

    # Test with default parameters (exclude_first=True)
    result = colormap_to_dict(colormap, num_colors=3)

    # Should have 3 colors plus None key
    assert len(result) == 4
    assert None in result
    assert result[None] == (0, 0, 0, 0)

    # Check that we have the expected keys (1, 2, 3 since exclude_first=True)
    assert 1 in result
    assert 2 in result
    assert 3 in result

    # Test with exclude_first=False
    result_no_exclude = colormap_to_dict(
        colormap, num_colors=3, exclude_first=False
    )
    assert len(result_no_exclude) == 4
    assert 1 in result_no_exclude
    assert 2 in result_no_exclude
    assert 3 in result_no_exclude

    # Test that colors are RGBA tuples
    for key, color in result.items():
        if key is not None:
            assert len(color) == 4  # RGBA
            assert all(isinstance(c, (int, float)) for c in color)

    # Test with different num_colors
    result_more_colors = colormap_to_dict(
        colormap, num_colors=5, exclude_first=True
    )
    assert len(result_more_colors) == 6  # 5 colors + None
    assert all(i in result_more_colors for i in [1, 2, 3, 4, 5])


def test_update_frequency_in_metadata(make_napari_viewer):
    """Test update_frequency_in_metadata function."""
    # Create a test image layer
    raw_flim_data = make_raw_flim_data(n_time_bins=10)
    intensity_layer = make_intensity_layer_with_phasors(raw_flim_data)

    # Test updating frequency when no settings exist
    test_frequency = 80.0
    if "settings" in intensity_layer.metadata:
        del intensity_layer.metadata["settings"]
    update_frequency_in_metadata(intensity_layer, test_frequency)
    assert "settings" in intensity_layer.metadata
    assert "frequency" in intensity_layer.metadata["settings"]
    assert intensity_layer.metadata["settings"]["frequency"] == test_frequency

    # Test updating frequency when settings already exist
    intensity_layer.metadata["settings"]["existing_setting"] = "test"
    new_frequency = 120.0
    update_frequency_in_metadata(intensity_layer, new_frequency)
    assert intensity_layer.metadata["settings"]["frequency"] == new_frequency
    assert intensity_layer.metadata["settings"]["existing_setting"] == "test"
