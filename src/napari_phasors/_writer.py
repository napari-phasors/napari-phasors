"""
This module contains functions to write images and phasor coordinates
to `OME-TIFF` format.

"""

from __future__ import annotations

import importlib.metadata
import json
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from napari.layers import Image
from phasorpy.io import phasor_to_ometiff

if TYPE_CHECKING:
    DataType = Union[Any, Sequence[Any]]
    FullLayerData = tuple[DataType, dict, str]


def _convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types.

    Handles numpy scalars in both dict keys and values, making
    the data structure safe for ``json.dumps``.

    Parameters
    ----------
    obj : any
        The object to convert. Can be a dict, list, tuple,
        numpy scalar, numpy array, or any other type.

    Returns
    -------
    any
        The converted object with all numpy types replaced
        by their native Python equivalents.
    """
    if isinstance(obj, dict):
        return {
            _convert_numpy_types(k): _convert_numpy_types(v)
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return type(obj)(_convert_numpy_types(item) for item in obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _extract_z_spacing_um(
    image_layer: Any,
    data: np.ndarray,
    metadata: dict,
) -> float | None:
    """Return z-spacing in micrometers when a Z axis is present."""
    if not hasattr(data, 'ndim') or data.ndim < 3:
        return None

    z_axis = None
    axis_labels = getattr(image_layer, 'axis_labels', None)
    if axis_labels is not None:
        for idx, label in enumerate(axis_labels):
            if str(label).strip().lower() == 'z':
                z_axis = idx
                break

    if z_axis is None:
        if data.ndim == 3 or metadata.get("stack_files"):
            z_axis = 0
        else:
            return None

    scale = getattr(image_layer, 'scale', None)
    if scale is not None and len(scale) > z_axis:
        try:
            value = float(scale[z_axis])
            if value > 0:
                return value
        except (TypeError, ValueError):
            pass

    settings = metadata.get("settings", {})
    fallback = settings.get("z_spacing_um")
    if fallback is None:
        return None

    try:
        value = float(fallback)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def write_ome_tiff(path: str, image_layer: Any) -> list[str]:
    """Save image layer with phasor coordinates as 'OME-TIFF'.

    For layers with phasor metadata, saves mean intensity and phasor coordinates.
    For layers without phasor metadata, saves the raw layer data as OME-TIFF.

    Parameters
    ----------
    path : str
        A string path indicating where to save the image file.
    image_layer : napari.layers.Image
        Napari image layer or a list with the mean intensity image as the
        first element and as second element a dict with `metadata` as key.
        The value associated to 'metadata' must be a dict containing
        `G_original`, `S_original`, and `harmonics` keys with NumPy arrays
        for phasor data, or just raw image data for non-phasor layers.

    Returns
    -------
    A list containing the string path to the saved file.
    """
    # Extract metadata depending on input type
    if isinstance(image_layer, Image):
        metadata = image_layer.metadata
        data = image_layer.data
    else:
        metadata = image_layer[0][1]["metadata"]
        data = image_layer[0][0]

    z_spacing_um = _extract_z_spacing_um(image_layer, data, metadata)

    # Check if layer has phasor data
    has_phasor_data = (
        "original_mean" in metadata
        and "G_original" in metadata
        and "S_original" in metadata
        and "harmonics" in metadata
    )

    if not path.endswith(".ome.tif"):
        path += ".ome.tif"

    dims = None
    if isinstance(image_layer, Image):
        labels = getattr(image_layer, 'axis_labels', None)
        if labels is not None and len(labels) == data.ndim:
            dims = "".join(str(label).upper()[0] for label in labels)
            if not dims.endswith(("YX", "YXS")):
                dims = None

    if not dims:
        if data.ndim == 2:
            dims = "YX"
        elif data.ndim == 3:
            dims = "ZYX"
        elif data.ndim == 4:
            dims = "TZYX"
        else:
            dims = "TZCYX"[-data.ndim :] if data.ndim <= 5 else None

    metadata_dict = {}

    if z_spacing_um is not None:
        metadata_dict['PhysicalSizeZ'] = z_spacing_um
        metadata_dict['PhysicalSizeZUnit'] = 'µm'

    if isinstance(image_layer, Image):
        scale = getattr(image_layer, 'scale', None)
        if scale is not None:
            y_idx, x_idx = -2, -1
            labels = getattr(image_layer, 'axis_labels', None)
            if labels is not None:
                for i, label in enumerate(labels):
                    if str(label).lower() == 'y':
                        y_idx = i
                    if str(label).lower() == 'x':
                        x_idx = i

            if len(scale) > abs(y_idx) and scale[y_idx] > 0:
                metadata_dict['PhysicalSizeY'] = float(scale[y_idx])
                metadata_dict['PhysicalSizeYUnit'] = 'µm'
            if len(scale) > abs(x_idx) and scale[x_idx] > 0:
                metadata_dict['PhysicalSizeX'] = float(scale[x_idx])
                metadata_dict['PhysicalSizeXUnit'] = 'µm'

    if has_phasor_data:
        # Export with phasor data
        mean = metadata["original_mean"]
        G = metadata["G_original"]
        S = metadata["S_original"]
        harmonics = metadata["harmonics"]
        if "settings" in metadata:
            settings = metadata["settings"].copy()
        else:
            settings = {}
        if "summed_signal" in metadata:
            summed = metadata["summed_signal"]
            settings["summed_signal"] = (
                summed.tolist() if hasattr(summed, 'tolist') else summed
            )
        if z_spacing_um is not None:
            settings["z_spacing_um"] = z_spacing_um

        # Convert NumPy arrays in selections to lists for JSON serialization
        if "selections" in settings:
            # Make a copy of selections dict to avoid mutating the layer's metadata
            settings["selections"] = settings["selections"].copy()
            if "manual_selections" in settings["selections"]:
                # Note: The following code is commented out to avoid saving large selection maps.
                # manual_selections = {}
                # for sel_id, sel_array in settings["selections"]["manual_selections"].items():
                #     manual_selections[sel_id] = (
                #         sel_array.tolist() if hasattr(sel_array, 'tolist') else sel_array
                #     )
                # settings["selections"]["manual_selections"] = manual_selections
                del settings["selections"]["manual_selections"]

        settings["version"] = str(importlib.metadata.version('napari-phasors'))
        settings = _convert_numpy_types(settings)
        description = json.dumps(
            {"napari_phasors_settings": json.dumps(settings)}
        )

        phasor_to_ometiff(
            path,
            mean,
            G,
            S,
            harmonic=harmonics,
            description=description,
            dims=dims,
            metadata=metadata_dict,
        )
    else:
        # Export without phasor data - just save the raw image data
        import tifffile

        # Prepare basic metadata
        settings = {}
        if "settings" in metadata:
            settings = metadata["settings"].copy()
        if z_spacing_um is not None:
            settings["z_spacing_um"] = z_spacing_um

        settings["version"] = str(importlib.metadata.version('napari-phasors'))
        settings = _convert_numpy_types(settings)
        description = json.dumps(
            {"napari_phasors_settings": json.dumps(settings)}
        )

        if dims:
            metadata_dict['axes'] = dims

        tifffile.imwrite(
            path,
            data,
            metadata=metadata_dict if metadata_dict else None,
            description=description,
        )

    return [path]


def export_layer_as_image(
    path: str,
    image_layer: Image,
    include_colorbar: bool = True,
    current_step: Sequence[int] | None = None,
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
    phasor data (G, S arrays) in its metadata:

    * If present, the phasor data is exported with coordinate columns
      describing the pixel positions for each harmonic.
    * Otherwise, the raw layer data is flattened into a table of
      coordinates and values.

    Parameters
    ----------
    path : str
        Output CSV file path.
    image_layer : napari.layers.Image
        Image layer whose data or phasor features will be exported.
    """
    has_phasor_data = (
        "G" in image_layer.metadata
        and "S" in image_layer.metadata
        and "harmonics" in image_layer.metadata
    )

    if has_phasor_data:
        G = image_layer.metadata["G"]
        S = image_layer.metadata["S"]
        G_original = image_layer.metadata.get("G_original", G)
        S_original = image_layer.metadata.get("S_original", S)
        harmonics = np.atleast_1d(image_layer.metadata["harmonics"])

        # Get spatial shape (last 2 dimensions)
        spatial_shape = G.shape[-2:]
        n_pixels = np.prod(spatial_shape)

        # Create coordinate arrays
        coords = np.unravel_index(np.arange(n_pixels), spatial_shape)

        # Build dataframe with phasor data
        rows = []
        for h_idx, harmonic in enumerate(harmonics):
            # Handle both 2D (single harmonic) and 3D (multiple harmonics) cases
            if G.ndim == 3:
                g_flat = G[h_idx].ravel()
                s_flat = S[h_idx].ravel()
                g_orig_flat = G_original[h_idx].ravel()
                s_orig_flat = S_original[h_idx].ravel()
            else:
                g_flat = G.ravel()
                s_flat = S.ravel()
                g_orig_flat = G_original.ravel()
                s_orig_flat = S_original.ravel()

            for px_idx in range(n_pixels):
                if not (np.isnan(g_flat[px_idx]) and np.isnan(s_flat[px_idx])):
                    row = {
                        'harmonic': harmonic,
                        'G': g_flat[px_idx],
                        'S': s_flat[px_idx],
                        'G_original': g_orig_flat[px_idx],
                        'S_original': s_orig_flat[px_idx],
                    }
                    for dim, coord in enumerate(coords):
                        row[f'dim_{dim}'] = coord[px_idx]
                    rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)
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
