


def make_raw_flim_data(n_time_bins=1000, shape=(2, 5), time_constants = [0.01, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50]):
    """Generate a synthetic FLIM data with exponential decay for each pixel.
    """
    import numpy as np
    n_samples = shape[0] * shape[1]   
    # make a 1d array exponential decay for each time constant
    raw_flim_data = np.moveaxis(np.array([np.exp(-(1/time_constant) * np.linspace(0, 10, n_time_bins)) for time_constant in time_constants]).reshape((*shape, n_time_bins)), -1, 0)
    return raw_flim_data

def make_intensity_layer_with_phasors(raw_flim_data, axis=0, harmonic=None, filename='FLIM data'):
    """Generate an intensity image layer with phasor features.
    """
    import numpy as np
    from phasorpy.phasor import phasor_from_signal
    from napari.layers import Labels, Image
    import pandas as pd
    if harmonic is None:
        harmonic = 1
    mean_intensity_image, G_image, S_image = phasor_from_signal(raw_flim_data, axis=axis, harmonic=harmonic)
    pixel_id = np.arange(1, mean_intensity_image.size + 1)
    if len(harmonic) > 1:
        table = pd.DataFrame([])
        for i in range(G_image.shape[0]):
            sub_table = pd.DataFrame({'label': pixel_id, 'Average Image': mean_intensity_image.ravel(), 'G': G_image[i].ravel(), 'S': S_image[i].ravel(), 'harmonic': harmonic[i]})
            table = pd.concat([table, sub_table])
    else:
        table = pd.DataFrame({'label': pixel_id, 'Average Image': mean_intensity_image.ravel(), 'G': G_image.ravel(), 'S': S_image.ravel(), 'harmonic': harmonic})
    labels_data = pixel_id.reshape(mean_intensity_image.shape)
    labels_layer = Labels(labels_data, name=filename + ' Phasor Features Layer', scale=(1, 1), features=table)
    mean_intensity_image_layer = Image(mean_intensity_image, name=filename + ' Intensity Image', metadata={'phasor_features_labels_layer': labels_layer})
    return mean_intensity_image_layer