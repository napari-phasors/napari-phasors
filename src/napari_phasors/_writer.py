"""
This module contains functions to write images and phasor coordinates
to `OME-TIFF` format.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Sequence, Tuple, Union

from phasorpy.io import write_ometiff_phasor
import numpy as np

if TYPE_CHECKING:
    DataType = Union[Any, Sequence[Any]]
    FullLayerData = Tuple[DataType, dict, str]

def write_ome_tiff(path: str, data: Any) -> List[str]:
    """Save Labels layer with phasor coordinates as 'OME-TIFF'.

    Parameters
    ----------
    path : str
        A string path indicating where to save the image file.
    data : The layer data
        The `.data` attribute from the napari layer.

    Notes
    -----
    Labels layer `data` must contain as first element the mean intensity
    image and as second element a `metadata` dict with
    `phasor_features_labels_layer` as key which contains phasor coordinates
    as a table with `G` and `S` columns.

    Returns
    -------
    A list containing the string path to the saved file.
    """
    phasor_data = data[0][1]['metadata']['phasor_features_labels_layer']
    mean = data[0][0]
    G = np.reshape(phasor_data.features['G'], mean.shape)
    S = np.reshape(phasor_data.features['S'], mean.shape)
    write_ometiff_phasor(path, mean, G, S)
    return [path]
