import numpy as np
from biaplotter.plotter import ArtistType

from napari_phasors._synthetic_generator import (
    make_intensity_layer_with_phasors,
    make_raw_flim_data,
)
from napari_phasors.plotter import PlotterWidget

# Create a synthetic FLIM data and an intensity image layer with phasors for testing
raw_flim_data = make_raw_flim_data()
harmonic = [1, 2, 3]
intensity_image_layer = make_intensity_layer_with_phasors(
    raw_flim_data, harmonic=harmonic
)


def test_phasor_plotter(make_napari_viewer):
    # Intialize viewer and add intensity image layer with phasors data
    viewer = make_napari_viewer()
    viewer.add_layer(intensity_image_layer)
    phasors_table = intensity_image_layer.metadata[
        "phasor_features_labels_layer"
    ].features
    assert phasors_table.shape == (
        30,
        5,
    )  # rows: 10 pixels (2x5 image) x 3 harmonics; columns: 5

    # Create Plotter widget
    plotter = PlotterWidget(viewer)

    # Check that plotter adds a new default manual selection column to the table
    phasors_table = intensity_image_layer.metadata[
        "phasor_features_labels_layer"
    ].features
    assert phasors_table.shape == (30, 6)
    assert "MANUAL SELECTION #1" in phasors_table.columns

    # Call the plot method
    plotter.plot()
    # Check that new layer is created
    assert len(viewer.layers) == 2
    # Check that 'Phasors Selected' Labels layer is empty
    assert viewer.layers["Phasors Selected"].data.any() == False

    # Check that Image layer with phasors is in the plotter combobox
    image_layer_combobox_items = [
        plotter.plotter_inputs_widget.image_layer_with_phasor_features_combobox.itemText(
            i
        )
        for i in range(
            plotter.plotter_inputs_widget.image_layer_with_phasor_features_combobox.count()
        )
    ]
    assert intensity_image_layer.name in image_layer_combobox_items

    # Update input parameters
    plotter.harmonic = 2
    plotter.histogram_colormap = "viridis"
    plotter.histogram_bins = 5
    plotter.histogram_log_scale = True
    plotter.plot_type = ArtistType.SCATTER.name
    plotter.plot()

    # Add new phasor selection id
    plotter.selection_id = "selection_1"
    # Check that table contains new column
    phasors_table = intensity_image_layer.metadata[
        "phasor_features_labels_layer"
    ].features
    # table now has 5 DATA columns + 2 SELECTION columns
    assert phasors_table.shape == (30, 7)
    assert "selection_1" in phasors_table.columns

    # Select first 3 points
    manual_selection = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    plotter.canvas_widget.active_artist.color_indices = manual_selection
    # Check that 'Phasors Selected' Labels layer contains selection
    assert (
        viewer.layers["Phasors Selected"].data.ravel().tolist()
        == manual_selection.tolist()
    )
    # Check that new column in phasors_table contains selection for each harmonic
    for h in harmonic:
        h_mask = phasors_table["harmonic"] == h
        assert (
            phasors_table.loc[h_mask, "selection_1"].values.tolist()
            == manual_selection.tolist()
        )
