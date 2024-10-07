import numpy as np
from phasorpy.phasor import phasor_filter, phasor_threshold

from napari_phasors._synthetic_generator import (
    make_intensity_layer_with_phasors,
    make_raw_flim_data,
)
from napari_phasors._utils import apply_filter_and_threshold


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
    original_g, original_s = phasor_filter(
        original_g, original_s, repeat=1, size=3, axes=(1, 2)
    )
    mean, original_g, original_s = phasor_threshold(
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
