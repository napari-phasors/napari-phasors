"""
This module contains functions to write images and phasor coordinates
to `OME-TIFF` format.

"""

from __future__ import annotations

import importlib.metadata
import json
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
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


def export_layer_as_image(
    path: str,
    image_layer: Image,
    include_colorbar: bool = True,
    current_step: Optional[Sequence[int]] = None,
) -> None:
    """Export an image layer as an image file using its colormap and contrast limits.

    The function extracts a 2D slice from the provided ``image_layer`` and saves it
    using Matplotlib, preserving the napari colormap and contrast limits. For
    multi-dimensional data, ``current_step`` is used to select the slice that
    matches the current viewer position.

    Parameters
    ----------
    path : str
        Output file path. The extension determines the image format (e.g. ``.png``,
        ``.jpg``, ``.tif``).
    image_layer : napari.layers.Image
        Image layer to export.
    include_colorbar : bool, optional
        If ``True``, include a colorbar in the exported figure. Default is ``True``.
    current_step : sequence of int, optional
        Indices corresponding to the current position along non-spatial
        dimensions, typically taken from ``viewer.dims.current_step``. If not
        provided and the data is multi-dimensional, the first index (0) is used
        for each non-spatial dimension.
    """
    data = image_layer.data

    if data.ndim > 2:
        if current_step is not None:
            step = list(current_step)
            indices = [
                step[i] if i < len(step) else 0 for i in range(data.ndim - 2)
            ]
        else:
            indices = [0] * (data.ndim - 2)

        slice_indices = tuple(indices) + (slice(None), slice(None))
        data_2d = data[slice_indices]
    else:
        data_2d = data

    napari_cmap = image_layer.colormap
    if hasattr(napari_cmap, "colors"):
        cmap = LinearSegmentedColormap.from_list(
            napari_cmap.name, napari_cmap.colors
        )
    else:
        try:
            cmap = plt.get_cmap(napari_cmap.name)
        except (AttributeError, ValueError):
            cmap = plt.get_cmap("gray")

    clim = image_layer.contrast_limits

    # Calculate aspect ratio from data shape
    height, width = data_2d.shape
    aspect_ratio = width / height

    base_height = 8
    base_width = base_height * aspect_ratio

    if include_colorbar:
        colorbar_width = 0.5  # Width for colorbar in inches
        total_width = base_width + colorbar_width
        fig, (ax, cax) = plt.subplots(
            1,
            2,
            figsize=(total_width, base_height),
            gridspec_kw={"width_ratios": [base_width, colorbar_width]},
            facecolor="black",
        )
    else:
        fig, ax = plt.subplots(
            figsize=(base_width, base_height), facecolor="black"
        )

    im = ax.imshow(
        data_2d,
        cmap=cmap,
        vmin=clim[0],
        vmax=clim[1],
        interpolation="nearest",
        aspect="auto",
    )
    ax.axis("off")

    if include_colorbar:
        cbar = plt.colorbar(im, cax=cax)
        cbar.ax.tick_params(colors="white")

    plt.tight_layout(pad=0)

    dpi = 300

    # For JPEG, set facecolor to white since it doesn't support transparency
    if path.endswith((".jpg", ".jpeg")):
        fig.patch.set_facecolor("white")
        if include_colorbar:
            cax.set_ylabel(
                "Intensity", rotation=270, labelpad=15, color="black"
            )
            cbar.ax.tick_params(colors="black")

    plt.savefig(
        path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.1,
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)


def export_layer_as_csv(path: str, image_layer: Image) -> None:
    """Export layer data or phasor features as a CSV file.

    The function has two behaviors depending on whether the layer has
    a ``"phasor_features_labels_layer"`` entry in its metadata:

    * If present, the associated phasor features table is exported with
      additional ``dim_*`` columns describing the pixel coordinates for
      each harmonic.
    * Otherwise, the raw layer data is flattened into a table of
      coordinates and values.

    Parameters
    ----------
    path : str
        Output CSV file path.
    image_layer : napari.layers.Image
        Image layer whose data or phasor features will be exported.
    """
    has_phasor_table = (
        "phasor_features_labels_layer" in image_layer.metadata
        and image_layer.metadata["phasor_features_labels_layer"] is not None
    )

    if has_phasor_table:
        phasor_table = image_layer.metadata[
            "phasor_features_labels_layer"
        ].features
        harmonics = np.unique(phasor_table["harmonic"])

        coords = np.unravel_index(
            np.arange(image_layer.data.size), image_layer.data.shape
        )

        coords = [np.tile(coord, len(harmonics)) for coord in coords]

        for dim, coord in enumerate(coords):
            phasor_table[f"dim_{dim}"] = coord

        phasor_table = phasor_table.dropna()
        phasor_table.to_csv(path, index=False)
    else:
        data = image_layer.data
        if data.ndim == 2:
            rows, cols = data.shape
            y_coords, x_coords = np.meshgrid(
                range(rows), range(cols), indexing="ij"
            )
            df = pd.DataFrame(
                {
                    "y": y_coords.ravel(),
                    "x": x_coords.ravel(),
                    "value": data.ravel(),
                }
            )
        else:
            coords = np.unravel_index(np.arange(data.size), data.shape)
            df_dict = {f"dim_{i}": coord for i, coord in enumerate(coords)}
            df_dict["value"] = data.ravel()
            df = pd.DataFrame(df_dict)

        df.to_csv(path, index=False)
