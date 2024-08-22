"""
This module contains functions to write images and phasor coordinates
to `OME-TIFF` format.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Sequence, Tuple, Union

import numpy as np
from phasorpy.io import write_ometiff_phasor

if TYPE_CHECKING:
    DataType = Union[Any, Sequence[Any]]
    FullLayerData = Tuple[DataType, dict, str]


def write_ome_tiff(path: str, image_layer: Any) -> List[str]:
    """Save Labels layer with phasor coordinates as 'OME-TIFF'.

    Parameters
    ----------
    path : str
        A string path indicating where to save the image file.
    image_layer : napari.layers.Image
        Napari image layer. Must contain as first element the mean
        intensity image and as second element a dict with `metadata` as key.
        The value associated to 'metadata' must be a dict with
        `phasor_features_labels_layer` as key which contains as value a Labels
        layer with a Dataframe with `G`, `S`  and 'harmonic' columns.

    Returns
    -------
    A list containing the string path to the saved file.
    """
    mean = image_layer[0][0]
    phasor_data = image_layer[0][1]["metadata"]["phasor_features_labels_layer"]
    num_harmonics = len(phasor_data.features["harmonic"].unique())
    G = np.reshape(phasor_data.features["G"], (num_harmonics, *mean.shape))
    S = np.reshape(phasor_data.features["S"], (num_harmonics, *mean.shape))
    mean = mean[np.newaxis, ...].repeat(num_harmonics, axis=0)
    if not path.endswith(".ome.tif"):
        path += ".ome.tif"
    write_ometiff_phasor(path, mean, G, S)
    return
