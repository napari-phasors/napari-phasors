import numpy as np
from napari.layers import Image
from phasorpy.phasor import phasor_from_signal


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
        The harmonic number(s) to calculate the phasor. If None, the default is [1, 2].
    name : str
        Name of the layer.

    Returns
    -------
    layer or layer_data : napari.layers.Image or tuple
        If viewer is provided, returns the created Image layer object.
        Otherwise, returns a tuple (data, kwargs) where data is the mean
        intensity image and kwargs contains metadata with phasor coordinates
        stored as numpy arrays.
    """
    if harmonic is None:
        harmonic = [1, 2]

    mean_intensity_image, G_image, S_image = phasor_from_signal(
        raw_flim_data, axis=axis, harmonic=harmonic
    )

    # Calculate summed signal along time axis
    axes_to_sum = tuple(
        i for i in range(len(raw_flim_data.shape)) if i != axis
    )
    summed_signal = np.sum(raw_flim_data, axis=axes_to_sum)

    mean_intensity_image_layer = Image(
        mean_intensity_image,
        name=name + " Intensity Image",
        metadata={
            "original_mean": mean_intensity_image.copy(),
            "settings": {},
            "summed_signal": (
                summed_signal.tolist()
                if hasattr(summed_signal, 'tolist')
                else summed_signal
            ),
            "G": G_image,
            "S": S_image,
            "G_original": G_image.copy(),
            "S_original": S_image.copy(),
            "harmonics": harmonic,
        },
    )
    return mean_intensity_image_layer
