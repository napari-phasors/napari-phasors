"""
This module contains functions to write images and phasor coordinates
to `OME-TIFF` format.

"""

from __future__ import annotations

import importlib.metadata
import json
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from napari.layers import Labels
from phasorpy.io import phasor_to_ometiff

from ._utils import show_activity_progress

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


def _get_export_path(
    path: str,
    layer_name: str,
    layers_to_export: list[Any],
    layer_names: list[str],
    default_ext: str = "",
) -> str:
    """Determine the file path for a layer, preventing overwrites during multi-layer exports."""
    import napari

    directory = os.path.dirname(path)
    full_basename = os.path.basename(path)

    # Determine extension and base name
    if full_basename.lower().endswith(".ome.tif"):
        base_name = full_basename[:-8]
        ext = ".ome.tif"
    elif full_basename.lower().endswith(".ome.tiff"):
        base_name = full_basename[:-9]
        ext = ".ome.tiff"
    else:
        base_name, ext = os.path.splitext(full_basename)
        if not ext and default_ext:
            ext = default_ext

    # Try to detect if we are saving multiple selected layers via the napari GUI
    viewer = napari.current_viewer()
    is_multi_save = False
    viewer_layer_names = []
    if viewer is not None:
        selected_layers = list(viewer.layers.selection)
        if len(selected_layers) > 1:
            is_multi_save = True
            viewer_layer_names = [layer.name for layer in selected_layers]

    if len(layers_to_export) > 1:
        user_provided_name = base_name not in layer_names
        if user_provided_name:
            filename = f"{base_name}_{layer_name}{ext}"
        else:
            filename = f"{layer_name}{ext}"
    elif is_multi_save and layer_name in viewer_layer_names:
        user_provided_name = base_name not in viewer_layer_names
        if user_provided_name:
            filename = f"{base_name}_{layer_name}{ext}"
        else:
            filename = f"{layer_name}{ext}"
    else:
        filename = f"{base_name}{ext}"

    return os.path.join(directory, filename)


def write_ome_tiff(
    path: str,
    image_layer: Any,
    export_masked: bool = False,
) -> list[str]:
    """Save layer with phasor coordinates as 'OME-TIFF'.

    For layers with phasor metadata, saves mean intensity and phasor coordinates.
    For layers without phasor metadata, saves the raw layer data as OME-TIFF.

    Parameters
    ----------
    path : str
        A string path indicating where to save the image file.
    image_layer : napari.layers.Layer or tuple or list
        Napari layer-like object (such as Image or Labels) or a list/tuple
        representing a napari writer layer-data tuple (with the layer data as
        the first element and a dict with `metadata` as the second element).
        The metadata must contain `G_original`, `S_original`, and `harmonics`
        keys with NumPy arrays for phasor data, or just raw layer data for
        non-phasor layers.

    Returns
    -------
    A list containing the string path to the saved file.
    """
    # Extract metadata depending on input type

    if isinstance(image_layer, list) and not hasattr(image_layer, 'data'):
        layers_to_export = image_layer
    else:
        layers_to_export = [image_layer]

    layer_names = []
    for idx, current_layer in enumerate(layers_to_export):
        if hasattr(current_layer, 'data') and hasattr(
            current_layer, 'metadata'
        ):
            layer_names.append(getattr(current_layer, 'name', f"layer_{idx}"))
        else:
            attributes = (
                current_layer[1]
                if isinstance(current_layer, (list, tuple))
                and len(current_layer) > 1
                else {}
            )
            layer_names.append(
                attributes.get("name", f"layer_{idx}")
                if isinstance(attributes, dict)
                else f"layer_{idx}"
            )

    saved_paths = []
    for i, current_layer in enumerate(layers_to_export):
        if hasattr(current_layer, 'data') and hasattr(
            current_layer, 'metadata'
        ):
            metadata = current_layer.metadata
            data = current_layer.data
            layer_name = getattr(current_layer, 'name', f"layer_{i}")
        else:
            metadata = current_layer[1].get("metadata", {})
            data = current_layer[0]
            layer_name = current_layer[1].get("name", f"layer_{i}")

        current_path = _get_export_path(
            path,
            layer_name,
            layers_to_export,
            layer_names,
            default_ext=".ome.tif",
        )

        z_spacing_um = _extract_z_spacing_um(current_layer, data, metadata)

        # Check if layer has phasor data
        has_phasor_data = (
            "original_mean" in metadata
            and "G_original" in metadata
            and "S_original" in metadata
            and "harmonics" in metadata
        )

        dims = None
        if hasattr(current_layer, 'axis_labels'):
            labels = getattr(current_layer, 'axis_labels', None)
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

        if hasattr(current_layer, 'scale'):
            scale = getattr(current_layer, 'scale', None)
            if scale is not None:
                y_idx, x_idx = -2, -1
                labels = getattr(current_layer, 'axis_labels', None)
                if labels is not None:
                    for i_label, label in enumerate(labels):
                        if str(label).lower() == 'y':
                            y_idx = i_label
                        if str(label).lower() == 'x':
                            x_idx = i_label

                if len(scale) > abs(y_idx) and scale[y_idx] > 0:
                    metadata_dict['PhysicalSizeY'] = float(scale[y_idx])
                    metadata_dict['PhysicalSizeYUnit'] = 'µm'
                if len(scale) > abs(x_idx) and scale[x_idx] > 0:
                    metadata_dict['PhysicalSizeX'] = float(scale[x_idx])
                    metadata_dict['PhysicalSizeXUnit'] = 'µm'

        pbr = show_activity_progress(
            desc=f"Saving OME-TIFF {layer_name}...", total=2
        )
        try:
            if has_phasor_data:
                # Export with phasor data
                mean = metadata["original_mean"]
                G = metadata["G_original"]
                S = metadata["S_original"]
                harmonics = metadata["harmonics"]

                if export_masked and "mask" in metadata:
                    mask_data = metadata["mask"]
                    invert = metadata.get("mask_invert", False)
                    mask_invalid = mask_data > 0 if invert else mask_data <= 0

                    mean = np.where(mask_invalid, np.nan, mean.copy())

                    if G.ndim > mask_invalid.ndim:
                        mask_invalid_expanded = mask_invalid[np.newaxis, ...]
                        G = np.where(mask_invalid_expanded, np.nan, G.copy())
                        S = np.where(mask_invalid_expanded, np.nan, S.copy())
                    else:
                        G = np.where(mask_invalid, np.nan, G.copy())
                        S = np.where(mask_invalid, np.nan, S.copy())

                if "settings" in metadata:
                    settings = metadata["settings"].copy()
                else:
                    settings = {}
                if "summed_signal" in metadata:
                    summed = metadata["summed_signal"]
                    settings["summed_signal"] = (
                        summed.tolist()
                        if hasattr(summed, 'tolist')
                        else summed
                    )
                if z_spacing_um is not None:
                    settings["z_spacing_um"] = z_spacing_um

                # Convert NumPy arrays in selections to lists for JSON serialization
                if "selections" in settings:
                    settings["selections"] = settings["selections"].copy()
                    if "manual_selections" in settings["selections"]:
                        del settings["selections"]["manual_selections"]

                settings["version"] = str(
                    importlib.metadata.version('napari-phasors')
                )
                settings = _convert_numpy_types(settings)
                description = json.dumps(
                    {"napari_phasors_settings": json.dumps(settings)}
                )
                pbr.set_description("Writing phasor data...")
                pbr.update(1)
                phasor_to_ometiff(
                    current_path,
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

                settings["version"] = str(
                    importlib.metadata.version('napari-phasors')
                )
                settings = _convert_numpy_types(settings)
                description = json.dumps(
                    {"napari_phasors_settings": json.dumps(settings)}
                )

                if dims:
                    metadata_dict['axes'] = dims

                if export_masked and "mask" in metadata:
                    mask_data = metadata["mask"]
                    invert = metadata.get("mask_invert", False)
                    mask_invalid = mask_data > 0 if invert else mask_data <= 0

                    if data.ndim > mask_invalid.ndim:
                        expanded_shape = [1] * (
                            data.ndim - mask_invalid.ndim
                        ) + list(mask_invalid.shape)
                        mask_invalid_expanded = mask_invalid.reshape(
                            expanded_shape
                        )
                        data = np.where(
                            mask_invalid_expanded, np.nan, data.copy()
                        )
                    else:
                        data = np.where(mask_invalid, np.nan, data.copy())

                tifffile.imwrite(
                    current_path,
                    data,
                    metadata=metadata_dict if metadata_dict else None,
                    description=description,
                )
            saved_paths.append(current_path)
        finally:
            pbr.close()
    return saved_paths


def export_layer_as_image(
    path: str,
    image_layer: Any,
    include_colorbar: bool = True,
    current_step: Sequence[int] | None = None,
    dpi: int = 300,
) -> list[str]:
    """Export an image or labels layer as an image file using its colormap and contrast limits.

    The function extracts a 2D slice from the provided ``image_layer`` and saves it
    using Matplotlib, preserving the napari colormap and contrast limits. For
    multi-dimensional data, ``current_step`` is used to select the slice that
    matches the current viewer position.

    Parameters
    ----------
    path : str
        Output file path. The extension determines the image format (e.g. ``.png``,
        ``.jpg``, ``.tif``).
    image_layer : Any
        Image or Labels layer to export, or layer data tuple from napari writer.
    include_colorbar : bool, optional
        If ``True``, include a colorbar in the exported figure. Default is ``True``.
    current_step : sequence of int, optional
        Indices corresponding to the current position along non-spatial
        dimensions, typically taken from ``viewer.dims.current_step``. If not
        provided and the data is multi-dimensional, the first index (0) is used
        for each non-spatial dimension.
    dpi : int, optional
        Resolution (dots per inch) used when rendering the figure. Default is
        ``300``.
    """

    if isinstance(image_layer, list) and not hasattr(image_layer, 'data'):
        layers_to_export = image_layer
    else:
        layers_to_export = [image_layer]

    layer_names = []
    for idx, current_layer in enumerate(layers_to_export):
        if hasattr(current_layer, 'data'):
            layer_names.append(getattr(current_layer, 'name', f"layer_{idx}"))
        else:
            attributes = (
                current_layer[1]
                if isinstance(current_layer, (list, tuple))
                and len(current_layer) > 1
                else {}
            )
            layer_names.append(
                attributes.get("name", f"layer_{idx}")
                if isinstance(attributes, dict)
                else f"layer_{idx}"
            )

    saved_paths = []

    for i, current_layer in enumerate(layers_to_export):
        if hasattr(current_layer, 'data'):
            data = current_layer.data
            is_labels = isinstance(current_layer, Labels)
            colormap = getattr(current_layer, 'colormap', None)
            contrast_limits = getattr(current_layer, 'contrast_limits', None)
            gamma = getattr(current_layer, 'gamma', 1.0)
            layer_name = getattr(current_layer, 'name', f"layer_{i}")
        else:
            data = (
                current_layer[0][0]
                if isinstance(current_layer[0], tuple)
                else current_layer[0]
            )
            attributes = current_layer[1] if len(current_layer) > 1 else {}
            is_labels = (
                current_layer[2] == 'labels'
                if len(current_layer) > 2
                else False
            )
            colormap = attributes.get('colormap', None)
            contrast_limits = attributes.get('contrast_limits', None)
            gamma = attributes.get('gamma', 1.0)
            layer_name = attributes.get('name', f"layer_{i}")

        _, ext = os.path.splitext(path)
        ext_fallback = ext if ext else ".png"
        current_path = _get_export_path(
            path,
            layer_name,
            layers_to_export,
            layer_names,
            default_ext=ext_fallback,
        )

        if data.ndim > 2:
            if current_step is not None:
                step = list(current_step)
                indices = [
                    step[idx] if idx < len(step) else 0
                    for idx in range(data.ndim - 2)
                ]
            else:
                indices = [0] * (data.ndim - 2)

            slice_indices = tuple(indices) + (slice(None), slice(None))
            data_2d = data[slice_indices]
        else:
            data_2d = data

        if is_labels:
            layer_include_colorbar = False
            if colormap is not None and hasattr(colormap, 'map'):
                try:
                    data_2d = colormap.map(data_2d)
                except Exception:  # noqa: BLE001
                    cmap = 'nipy_spectral'
                    clim = [0, data_2d.max() if data_2d.max() > 0 else 1]
            else:
                cmap = 'nipy_spectral'
                clim = [0, data_2d.max() if data_2d.max() > 0 else 1]
        else:
            layer_include_colorbar = include_colorbar
            napari_cmap = colormap
            if isinstance(napari_cmap, dict):
                cmap_name = napari_cmap.get("name", "gray")
                cmap_colors = napari_cmap.get("colors", None)
                if cmap_colors is not None:
                    try:
                        cmap = LinearSegmentedColormap.from_list(
                            cmap_name, cmap_colors
                        )
                    except Exception:  # noqa: BLE001
                        try:
                            cmap = plt.get_cmap(cmap_name)
                        except Exception:  # noqa: BLE001
                            cmap = plt.get_cmap("gray")
                else:
                    try:
                        cmap = plt.get_cmap(cmap_name)
                    except Exception:  # noqa: BLE001
                        cmap = plt.get_cmap("gray")
            elif hasattr(napari_cmap, "colors"):
                cmap_name = getattr(napari_cmap, "name", "custom") or "custom"
                cmap = LinearSegmentedColormap.from_list(
                    cmap_name, napari_cmap.colors
                )
            else:
                try:
                    name = getattr(napari_cmap, "name", napari_cmap) or "gray"
                    cmap = plt.get_cmap(name)
                except (AttributeError, ValueError, TypeError):
                    cmap = plt.get_cmap("gray")
            clim = (
                contrast_limits
                if contrast_limits is not None
                else [data_2d.min(), data_2d.max()]
            )

        # Calculate aspect ratio from data shape
        height, width = data_2d.shape[:2]
        aspect_ratio = width / height

        base_height = 8
        base_width = base_height * aspect_ratio

        if layer_include_colorbar:
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

        imshow_kwargs = {
            "interpolation": "nearest",
            "aspect": "auto",
        }
        if data_2d.ndim == 2:
            imshow_kwargs["cmap"] = cmap
            if gamma is not None and gamma != 1.0:
                # Reproduce napari's rendering: normalize to the contrast
                # limits, then apply gamma via a power-law norm.
                imshow_kwargs["norm"] = PowerNorm(
                    gamma, vmin=clim[0], vmax=clim[1]
                )
            else:
                imshow_kwargs["vmin"] = clim[0]
                imshow_kwargs["vmax"] = clim[1]

        im = ax.imshow(data_2d, **imshow_kwargs)
        ax.axis("off")

        if layer_include_colorbar:
            cbar = plt.colorbar(im, cax=cax)
            cbar.ax.tick_params(colors="white")

        plt.tight_layout(pad=0)

        # For JPEG, set facecolor to white since it doesn't support transparency
        if current_path.endswith((".jpg", ".jpeg")):
            fig.patch.set_facecolor("white")
            if layer_include_colorbar:
                cax.set_ylabel(
                    "Intensity", rotation=270, labelpad=15, color="black"
                )
                cbar.ax.tick_params(colors="black")

        plt.savefig(
            current_path,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.1,
            facecolor=fig.get_facecolor(),
        )
        plt.close(fig)
        saved_paths.append(current_path)

    return saved_paths


def export_layer_as_csv(path: str, image_layer: Any) -> list[str]:
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
    image_layer : Any
        Image or Labels layer to export, or layer data tuple from napari writer.
    """

    if isinstance(image_layer, list) and not hasattr(image_layer, 'data'):
        layers_to_export = image_layer
    else:
        layers_to_export = [image_layer]

    layer_names = []
    for idx, current_layer in enumerate(layers_to_export):
        if hasattr(current_layer, 'data'):
            layer_names.append(getattr(current_layer, 'name', f"layer_{idx}"))
        else:
            attributes = (
                current_layer[1]
                if isinstance(current_layer, (list, tuple))
                and len(current_layer) > 1
                else {}
            )
            layer_names.append(
                attributes.get("name", f"layer_{idx}")
                if isinstance(attributes, dict)
                else f"layer_{idx}"
            )

    saved_paths = []

    for i, current_layer in enumerate(layers_to_export):
        if hasattr(current_layer, 'data'):
            data = current_layer.data
            metadata = getattr(current_layer, 'metadata', {})
            layer_name = getattr(current_layer, 'name', f"layer_{i}")
        else:
            data = current_layer[0]
            metadata = current_layer[1].get("metadata", {})
            layer_name = current_layer[1].get("name", f"layer_{i}")

        current_path = _get_export_path(
            path,
            layer_name,
            layers_to_export,
            layer_names,
            default_ext=".csv",
        )

        has_phasor_data = (
            "G" in metadata and "S" in metadata and "harmonics" in metadata
        )

        if has_phasor_data:
            G = metadata["G"]
            S = metadata["S"]
            G_original = metadata.get("G_original", G)
            S_original = metadata.get("S_original", S)
            harmonics = np.atleast_1d(metadata["harmonics"])

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
                    if not (
                        np.isnan(g_flat[px_idx]) and np.isnan(s_flat[px_idx])
                    ):
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
            df.to_csv(current_path, index=False)
        else:
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
                df_dict = {
                    f"dim_{i_dim}": coord for i_dim, coord in enumerate(coords)
                }
                df_dict["value"] = data.ravel()
                df = pd.DataFrame(df_dict)

            df.to_csv(current_path, index=False)
        saved_paths.append(current_path)
    return saved_paths
