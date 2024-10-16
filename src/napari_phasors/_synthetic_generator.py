def make_raw_flim_data(
    n_time_bins=1000,
    shape=(2, 5),
    time_constants=[0.01, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50],
    laser_frequency=40.0,
):
    """Generate a synthetic FLIM data with exponential decay for each pixel.

    Parameters
    ----------
    n_time_bins : int
        Number of time bins.
    shape : tuple (int, int)
        Shape of the synthetic FLIM data.
    time_constants : list of float
        Time constants for the exponential decay.
    laser_frequency : float
        Frequency of the laser in MHz

    Returns
    -------
    raw_flim_data : np.ndarray
        A synthetic FLIM data with exponential decay for each pixel. The shape of the array is (n_time_bins, shape[0], shape[1]).
    """
    import numpy as np

    n_pixels = np.prod(shape)
    # Ensure time_constants length matches the number of samples by repeating the whole list
    time_constants = np.tile(
        time_constants, int(np.ceil(n_pixels / len(time_constants)))
    )
    time_constants = time_constants[:n_pixels]
    # Make time array
    time_window_ns = 1e9 / (laser_frequency * 1e6)
    time_step = time_window_ns / n_time_bins  # ns
    time_array = np.arange(0, n_time_bins) * time_step
    # Make a 1d array exponential decay for each time constant
    raw_flim_data = np.moveaxis(
        np.array(
            [
                np.exp(-(1 / time_constant) * time_array)
                for time_constant in time_constants
            ]
        ).reshape((*shape, n_time_bins)),
        -1,
        0,
    )
    return raw_flim_data


def make_intensity_layer_with_phasors(
    raw_flim_data, axis=0, harmonic=None, name="FLIM data"
):
    """Generate an intensity image layer with phasor features.

    Parameters
    ----------
    raw_flim_data : np.ndarray
        A synthetic FLIM data with exponential decay for each pixel.
    axis : int
        Axis of the time bins.
    harmonic : int or list of int
        The harmonic number(s) to calculate the phasor. If None, the default is 1.
    name : str
        Name of the layer.

    Returns
    -------
    mean_intensity_image_layer : napari.layers.Image
        A napari Image layer with a Labels layer in the metadata, which stores the phasors table in its features attribute.
    """
    import numpy as np
    import pandas as pd
    from napari.layers import Image, Labels
    from phasorpy.phasor import phasor_from_signal

    if harmonic is None:
        harmonic = 1
    mean_intensity_image, G_image, S_image = phasor_from_signal(
        raw_flim_data, axis=axis, harmonic=harmonic
    )
    pixel_id = np.arange(1, mean_intensity_image.size + 1)
    if len(harmonic) > 1:
        table = pd.DataFrame([])
        for i in range(G_image.shape[0]):
            sub_table = pd.DataFrame(
                {
                    "label": pixel_id,
                    # "Average Image": mean_intensity_image.ravel(),
                    "G_original": G_image[i].ravel(),
                    "S_original": S_image[i].ravel(),
                    "G": G_image[i].ravel(),
                    "S": S_image[i].ravel(),
                    "harmonic": harmonic[i],
                }
            )
            table = pd.concat([table, sub_table])
    else:
        table = pd.DataFrame(
            {
                "label": pixel_id,
                # "Average Image": mean_intensity_image.ravel(),
                "G_original": G_image[i].ravel(),
                "S_original": S_image[i].ravel(),
                "G": G_image.ravel(),
                "S": S_image.ravel(),
                "harmonic": harmonic,
            }
        )
    labels_data = pixel_id.reshape(mean_intensity_image.shape)
    labels_layer = Labels(
        labels_data,
        name=name + " Phasor Features Layer",
        scale=(1, 1),
        features=table,
    )
    mean_intensity_image_layer = Image(
        mean_intensity_image,
        name=name + " Intensity Image",
        metadata={
            "phasor_features_labels_layer": labels_layer,
            "original_mean": mean_intensity_image,
        },
    )
    return mean_intensity_image_layer
