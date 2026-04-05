# FLIM Phasor Workflow

This guide walks through a complete FLIM (Fluorescence Lifetime Imaging Microscopy) phasor analysis workflow using napari-phasors.

## 1. Load your FLIM data

napari-phasors supports a wide range of FLIM file formats directly:

**FLIM file formats:**
- **FBD** (`.fbd`) — FlimBox data
- **SDT** (`.sdt`) — Becker & Hickl
- **PTU** (`.ptu`) — PicoQuant
- **OME-TIF** (`.ome.tif`, `.ome.tiff`) — Bio-Formats / OME
- **TIFF** (`.tif`, `.tiff`) — Generic TIFF stacks
- **BH** (`.bh`, `.b&h`) — Becker & Hickl binary
- **BHZ** (`.bhz`) — Becker & Hickl zipped
- **BIN** (`.bin`) — PicoQuant binary
- **FLIF** (`.flif`) — FLIM Labs
- **R64** (`.r64`), **REF** (`.ref`) — SimFCS referenced
- **IFLI** (`.ifli`) — FLIM Labs intensity
- **JSON** (`.json`) — FLIM Labs/FLIM processed

All formats above can be opened via **File → Open File(s)** or by drag-and-drop into napari.
> [!TIP]
> You can select and open multiple raw data files simultaneously. napari-phasors will automatically recognize and stack compatible files into a single 3D image.

For advanced import options and stack building with the custom widget, see
{doc}`custom_import`.

Use **File → Open File(s)** or drag-and-drop your file into napari.
> [!TIP]
> You can select and open multiple raw data files simultaneously. napari-phasors will automatically recognize and stack compatible files into a single 3D image.

Alternatively, load the built-in sample data: **File → Open Sample → napari-phasors**.

## 2. Compute phasors

Once a FLIM file is loaded, the phasor transform is computed automatically and stored in the layer metadata. Open the **Phasor Plot** widget via **Plugins → napari-phasors → Phasor Plot**.

![phasors_flim](https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/flim%20plot.gif)

## 3. Select layers

Multiple image layers can be selected simultaneously from the layer dropdown. Use **All** / **None** to quickly select or deselect all layers. A **primary layer** can be designated to drive plot settings.

![multiple_layers](https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/multiple%20layers.gif)

## 4. Calibrate

If you have a reference image of a known fluorophore, calibrate your data in the **Calibration** tab. See {doc}`calibration` for details.

## 5. Filter and threshold

Use the **Filter** tab to apply median or wavelet filters, and set automatic thresholds (Otsu, Li, Yen) to remove background noise. See {doc}`filtering_thresholding` for details.

## 6. Select regions of interest

Use the **Selection** tab to identify regions in phasor space using circular cursors, manual drawing, or automatic clustering. See {doc}`selections` for details.

## 7. Analyze phasor outputs

Run **Component Analysis** when you want to decompose mixtures into two or
more components and generate fraction maps. See {doc}`components` for details.

Use the **Phasor Mapping** tab to colormap each pixel by its apparent lifetime,
phasor phase, or phasor modulation. An interactive 1D histogram and statistics
table update automatically. See {doc}`phasor_mapping` and
{doc}`histogram_statistics` for details.

For practical visualization examples and plot customization tools, see
{doc}`plot_customization`.

## 8. Export results

Export your phasor coordinates, selections, and analysis results as OME-TIF or
CSV. Multiple layers can be exported simultaneously. See {doc}`exporting` for
details.
