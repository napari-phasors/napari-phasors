from src.napari_phasors._synthetic_generator import (
    make_intensity_layer_with_phasors,
    make_raw_flim_data,
)
from src.napari_phasors.plotter import PlotterWidget


if __name__ == "__main__":
    import napari

    time_constants = [0.1, 1, 2, 3, 4, 5, 10]
    raw_flim_data = make_raw_flim_data(time_constants=time_constants)
    harmonic = [1, 2, 3]
    # harmonic = [1]
    intensity_image_layer = make_intensity_layer_with_phasors(
        raw_flim_data, harmonic=harmonic
    )
    viewer = napari.Viewer()
    viewer.add_layer(intensity_image_layer)
    plotter = PlotterWidget(viewer)
    viewer.window.add_dock_widget(plotter, area="right")
    napari.run()