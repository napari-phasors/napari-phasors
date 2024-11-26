"""
This module contains functions to write images and phasor coordinates
to `OME-TIFF` format.

"""

from __future__ import annotations

import importlib.metadata
import json
from typing import TYPE_CHECKING, Any, List, Sequence, Tuple, Union

import numpy as np
from napari.layers import Image
from phasorpy.io import phasor_to_ometiff

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
        Napari image layer or a list with the mean intensity image as the
        first element and as second element a dict with `metadata` as key.
        The value associated to 'metadata' must be a dict with
        `phasor_features_labels_layer` as key which contains as value a Labels
        layer with a Dataframe with `G`, `S`  and 'harmonic' columns.

    Returns
    -------
    A list containing the string path to the saved file.
    """
    if isinstance(image_layer, Image):
        mean = image_layer.metadata["original_mean"]
        phasor_data = image_layer.metadata["phasor_features_labels_layer"]
        if "settings" in image_layer.metadata:
            settings = image_layer.metadata["settings"]
        else:
            settings = {}
    else:
        mean = image_layer[0][1]["metadata"]["original_mean"]
        phasor_data = image_layer[0][1]["metadata"][
            "phasor_features_labels_layer"
        ]
        if "settings" in image_layer[0][1]["metadata"]:
            settings = image_layer[0][1]["settings"]
        else:
            settings = {}
    harmonics = phasor_data.features["harmonic"].unique()
    G = np.reshape(
        phasor_data.features["G_original"], (len(harmonics), *mean.shape)
    )
    S = np.reshape(
        phasor_data.features["S_original"], (len(harmonics), *mean.shape)
    )
    if not path.endswith(".ome.tif"):
        path += ".ome.tif"
    settings["version"] = str(importlib.metadata.version('napari-phasors'))
    description = json.dumps({"napari_phasors_settings": json.dumps(settings)})
    phasor_to_ometiff(
        path,
        mean,
        G,
        S,
        harmonic=harmonics,
        description=description,
    )
    return path
