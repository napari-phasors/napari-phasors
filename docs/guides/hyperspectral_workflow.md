# Hyperspectral Phasor Workflow

This guide walks through a complete hyperspectral imaging phasor analysis workflow using napari-phasors.

## 1. Load your hyperspectral data

napari-phasors supports several hyperspectral file formats:

**Hyperspectral file formats:**
- **CZI** (`.czi`) — Carl Zeiss Image
- **LSM** (`.lsm`) — Zeiss LSM
- **LIF** (`.lif`) — Leica Image File
- **OME-TIF** (`.ome.tif`, `.ome.tiff`) — Bio-Formats / OME
- **TIFF** (`.tif`, `.tiff`) — Generic TIFF stacks

All formats above can be opened via **File → Open File(s)** or by drag-and-drop into napari.
> [!TIP]
> You can select and open multiple raw data files simultaneously. napari-phasors will automatically recognize and stack compatible files into a single 3D image.

Alternatively, load the built-in sample data: **File → Open Sample → napari-phasors → Paramecium**.

For advanced import options and stack building with the custom widget, see
{doc}`open_files`.

## 2. Compute phasors

Once a hyperspectral file is loaded, the phasor transform is computed automatically and stored in the layer metadata. Open the **Phasor Plot** widget via **Plugins → napari-phasors → Phasor Plot**.

To visualize hyperspectral data in the full universal circle, uncheck **Universal Semi-Circle/Full Polar Plot** in the **Plot Settings** tab.

<video width="100%" autoplay loop muted playsinline poster="https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/hsi%20plot.gif">
  <source src="https://github.com/napari-phasors/napari-phasors-data/raw/main/videos/hsi%20plot.mp4" type="video/mp4">
</video>

## 3. Select layers

Multiple image layers can be selected simultaneously from the layer dropdown. Use **All** / **None** to quickly select or deselect all layers. A **primary layer** can be designated to drive plot settings. See {doc}`multi_layer_analysis` for details.


## 4. Filter and threshold

Use the **Filter** tab to apply median or wavelet filters, and set automatic thresholds (Otsu, Li, Yen) to remove background noise. See {doc}`filtering_thresholding` for details.

## 5. Select regions of interest

Use the **Selection** tab to identify regions in phasor space using circular cursors, manual drawing, or automatic clustering. See {doc}`phasor_selection` for details.

In hyperspectral analysis, positions in the phasor plot relate to **spectral composition** rather than fluorescence lifetimes.

## 6. Analyze phasor outputs

Run **Component Analysis** when you want to decompose mixtures into two or more components and generate fraction maps. See {doc}`component_analysis` for details.

Use the **Phasor Mapping** tab to colormap each pixel by its apparent spectral center (phase) or spectral width (modulation). An interactive 1D histogram and statistics table update automatically. See {doc}`phasor_mapping` and {doc}`histogram_statistics` for details.

For practical visualization examples and plot customization tools, see
{doc}`plot_customization`.

## 7. Export results

Export your phasor coordinates, selections, and analysis results as OME-TIF or CSV. Multiple layers can be exported simultaneously. See {doc}`exporting` for details.
