def test_synthetic_generator():
    import numpy as np

    from napari_phasors._synthetic_generator import (
        make_intensity_layer_with_phasors,
        make_raw_flim_data,
    )

    # Create a synthetic FLIM data and an intensity image layer with phasors
    raw_flim_data = make_raw_flim_data()
    harmonic = [1, 2, 3]
    intensity_image_layer = make_intensity_layer_with_phasors(
        raw_flim_data, harmonic=harmonic
    )

    # Check that Image layer contains phasor data in metadata
    assert "G" in intensity_image_layer.metadata
    assert "S" in intensity_image_layer.metadata
    assert "G_original" in intensity_image_layer.metadata
    assert "S_original" in intensity_image_layer.metadata
    assert "harmonics" in intensity_image_layer.metadata
    assert "summed_signal" in intensity_image_layer.metadata
    assert "original_mean" in intensity_image_layer.metadata

    # Check that phasor arrays have correct shape (n_harmonics, height, width)
    G = intensity_image_layer.metadata["G"]
    S = intensity_image_layer.metadata["S"]
    harmonics = intensity_image_layer.metadata["harmonics"]
    
    assert isinstance(G, np.ndarray)
    assert isinstance(S, np.ndarray)
    assert G.shape == (len(harmonic), *raw_flim_data.shape[1:])
    assert S.shape == (len(harmonic), *raw_flim_data.shape[1:])
    assert harmonics == harmonic
