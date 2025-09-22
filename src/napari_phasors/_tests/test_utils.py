import numpy as np
from matplotlib.colors import ListedColormap
from phasorpy.phasor import (
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
    original_g = intensity_image_layer.metadata[
        'phasor_features_labels_layer'
    ].features['G']
    original_s = intensity_image_layer.metadata[
        'phasor_features_labels_layer'
    ].features['S']
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
    phasor_features = intensity_image_layer.metadata[
        'phasor_features_labels_layer'
    ].features
    assert np.all(phasor_features['G_original'] == original_g)
    assert np.all(phasor_features['S_original'] == original_s)
    harmonics = np.unique(phasor_features['harmonic'])
    original_g = np.reshape(
        original_g, (len(harmonics),) + original_mean.shape
    )
    original_s = np.reshape(
        original_s, (len(harmonics),) + original_mean.shape
    )
    original_mean, original_g, original_s = phasor_filter_median(
        original_mean, original_g, original_s, repeat=1, size=3
    )
    _, original_g, original_s = phasor_threshold(
        original_mean, original_g, original_s, threshold
    )
    filtered_thresholded_g = np.reshape(
        phasor_features['G'], (len(harmonics),) + original_mean.shape
    )
    filtered_thresholded_s = np.reshape(
        phasor_features['S'], (len(harmonics),) + original_mean.shape
    )
    assert np.allclose(original_g, filtered_thresholded_g, equal_nan=True)
    assert np.allclose(original_s, filtered_thresholded_s, equal_nan=True)

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
    original_g = (
        intensity_image_layer.metadata['phasor_features_labels_layer']
        .features['G_original']
        .copy()
    )
    original_s = (
        intensity_image_layer.metadata['phasor_features_labels_layer']
        .features['S_original']
        .copy()
    )

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

    # Check that phasor features are updated (they should be different from original)
    phasor_features = intensity_image_layer.metadata[
        'phasor_features_labels_layer'
    ].features
    assert np.all(phasor_features['G_original'] == original_g)
    assert np.all(phasor_features['S_original'] == original_s)

    # The filtered G and S should be different from original (unless no filtering occurred)
    # We can't easily predict the exact values, but we can check they exist and are valid
    assert 'G' in phasor_features
    assert 'S' in phasor_features
    assert len(phasor_features['G']) == len(original_g)
    assert len(phasor_features['S']) == len(original_s)

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
    original_g = (
        intensity_image_layer.metadata['phasor_features_labels_layer']
        .features['G_original']
        .copy()
    )
    original_s = (
        intensity_image_layer.metadata['phasor_features_labels_layer']
        .features['S_original']
        .copy()
    )

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
    phasor_features = intensity_image_layer.metadata[
        'phasor_features_labels_layer'
    ].features

    # The G and S values should be thresholded versions of the original
    # (without wavelet filtering)
    harmonics = np.unique(phasor_features['harmonic'])
    reshaped_g = np.reshape(
        original_g, (len(harmonics),) + original_mean.shape
    )
    reshaped_s = np.reshape(
        original_s, (len(harmonics),) + original_mean.shape
    )

    # Apply only threshold (no filtering) for comparison
    _, expected_g, expected_s = phasor_threshold(
        original_mean, reshaped_g, reshaped_s, threshold
    )

    actual_g = np.reshape(
        phasor_features['G'], (len(harmonics),) + original_mean.shape
    )
    actual_s = np.reshape(
        phasor_features['S'], (len(harmonics),) + original_mean.shape
    )

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
    original_g = (
        intensity_image_layer.metadata['phasor_features_labels_layer']
        .features['G_original']
        .copy()
    )
    original_s = (
        intensity_image_layer.metadata['phasor_features_labels_layer']
        .features['S_original']
        .copy()
    )

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
    expected_g = np.reshape(
        original_g, (len(custom_harmonics),) + original_mean.shape
    )
    expected_s = np.reshape(
        original_s, (len(custom_harmonics),) + original_mean.shape
    )

    expected_mean, expected_g, expected_s = phasor_filter_pawflim(
        original_mean,
        expected_g,
        expected_s,
        sigma=sigma,
        levels=levels,
        harmonic=custom_harmonics,
    )
    # Apply threshold to get the final expected values (including mean modifications)
    expected_mean, expected_g, expected_s = phasor_threshold(
        expected_mean, expected_g, expected_s, threshold
    )

    # Compare results
    phasor_features = intensity_image_layer.metadata[
        'phasor_features_labels_layer'
    ].features
    actual_g = np.reshape(
        phasor_features['G'], (len(custom_harmonics),) + original_mean.shape
    )
    actual_s = np.reshape(
        phasor_features['S'], (len(custom_harmonics),) + original_mean.shape
    )

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
    original_g = (
        intensity_image_layer.metadata['phasor_features_labels_layer']
        .features['G_original']
        .copy()
    )
    original_s = (
        intensity_image_layer.metadata['phasor_features_labels_layer']
        .features['S_original']
        .copy()
    )

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
    phasor_features = intensity_image_layer.metadata[
        'phasor_features_labels_layer'
    ].features

    harmonics = np.unique(phasor_features['harmonic'])
    reshaped_g = np.reshape(
        original_g, (len(harmonics),) + original_mean.shape
    )
    reshaped_s = np.reshape(
        original_s, (len(harmonics),) + original_mean.shape
    )

    # Apply only threshold for comparison
    _, expected_g, expected_s = phasor_threshold(
        original_mean, reshaped_g, reshaped_s, threshold
    )

    actual_g = np.reshape(
        phasor_features['G'], (len(harmonics),) + original_mean.shape
    )
    actual_s = np.reshape(
        phasor_features['S'], (len(harmonics),) + original_mean.shape
    )

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

    # Get original features
    original_g = (
        intensity_image_layer.metadata['phasor_features_labels_layer']
        .features['G_original']
        .copy()
    )
    original_s = (
        intensity_image_layer.metadata['phasor_features_labels_layer']
        .features['S_original']
        .copy()
    )

    # Manually create a subset of G and S for custom harmonics
    # This simulates what should happen when using a subset of harmonics
    layer_harmonics = np.unique(
        intensity_image_layer.metadata[
            'phasor_features_labels_layer'
        ].features['harmonic']
    )

    # Create new features with only harmonic [1]
    phasor_features = intensity_image_layer.metadata[
        'phasor_features_labels_layer'
    ]
    mask = phasor_features.features['harmonic'] == 1

    # Create new G_original and S_original with only harmonic 1
    new_g_original = original_g[mask]
    new_s_original = original_s[mask]
    new_harmonics = phasor_features.features['harmonic'][mask]

    # Update the layer's features to have only harmonic 1
    phasor_features.features = {
        'G_original': new_g_original,
        'S_original': new_s_original,
        'G': new_g_original.copy(),
        'S': new_s_original.copy(),
        'harmonic': new_harmonics,
    }

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

    # Should work without error since harmonics match the modified features
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
