def test_synthetic_generator():
    from napari.layers import Labels

    from napari_phasors._synthetic_generator import (
        make_intensity_layer_with_phasors,
        make_raw_flim_data,
    )
    from napari_phasors.selection_tab import DATA_COLUMNS

    # Create a synthetic FLIM data and an intensity image layer with phasors
    raw_flim_data = make_raw_flim_data()
    harmonic = [1, 2, 3]
    intensity_image_layer = make_intensity_layer_with_phasors(
        raw_flim_data, harmonic=harmonic
    )

    # Check that Image layer contains Labels layer in metadata
    assert "phasor_features_labels_layer" in intensity_image_layer.metadata
    assert isinstance(
        intensity_image_layer.metadata["phasor_features_labels_layer"], Labels
    )
    assert hasattr(
        intensity_image_layer.metadata["phasor_features_labels_layer"],
        "features",
    )

    # Check phasors_table generation
    phasors_table = intensity_image_layer.metadata[
        "phasor_features_labels_layer"
    ].features
    # Ensure all items in DATA_COLUMNS are columns in the phasors_table
    for column in DATA_COLUMNS:
        assert column in phasors_table
