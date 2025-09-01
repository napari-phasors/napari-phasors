import numpy as np
from matplotlib.colors import ListedColormap
from phasorpy.phasor import phasor_filter_median, phasor_threshold

from napari_phasors._synthetic_generator import (
    make_intensity_layer_with_phasors,
    make_raw_flim_data,
)
from napari_phasors._utils import (
    apply_filter_and_threshold,
    colormap_to_dict,
    update_frequency_in_metadata,
)


def test_apply_filter_and_threshold(make_napari_viewer):
    """Test apply_filter_and_threshold function."""
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
    apply_filter_and_threshold(intensity_image_layer, threshold=threshold)
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
        original_g, (len(harmonics),) + original_mean.data.shape
    )
    original_s = np.reshape(
        original_s, (len(harmonics),) + original_mean.data.shape
    )
    original_mean, original_g, original_s = phasor_filter_median(
        original_mean, original_g, original_s, repeat=1, size=3
    )
    _, original_g, original_s = phasor_threshold(
        original_mean, original_g, original_s, threshold
    )
    filtered_thresholded_g = np.reshape(
        phasor_features['G'], (len(harmonics),) + original_mean.data.shape
    )
    filtered_thresholded_s = np.reshape(
        phasor_features['S'], (len(harmonics),) + original_mean.data.shape
    )
    assert np.allclose(original_g, filtered_thresholded_g, equal_nan=True)
    assert np.allclose(original_s, filtered_thresholded_s, equal_nan=True)


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
