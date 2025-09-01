#%%
import napari
from napari_phasors._sample_data import paramecium_HSI_sample_data
from napari_phasors.plotter import PlotterWidget  # Replace with actual widget import
from napari.layers import Image
if __name__ == "__main__":
    # Load sample data (assuming it returns a tuple: (data, metadata))
    sample = paramecium_HSI_sample_data()
    viewer = napari.Viewer()
    layer = Image(
        sample[0][0],
        name='Paramecium HSI Sample Data',
        metadata=sample[0][1]['metadata']
    )
    viewer.add_layer(layer)

    # # Add your plotter widget (adjust as needed)
    viewer.window.add_dock_widget(PlotterWidget(viewer))

    napari.run()
# %%
