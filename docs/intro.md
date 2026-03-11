(intro)=

# napari-phasors

napari-phasors is a comprehensive plugin, based on the [phasorpy](https://www.phasorpy.org/) library, that provides a complete workflow for phasor analysis in [napari](https://napari.org). It includes widgets for:

- **Reading** various FLIM and hyperspectral file formats (`.fbd`, `.sdt`, `.ptu`, `.ome.tif`)
- **Phasor analysis** on multiple layers simultaneously, with mask support for region-of-interest restriction
- **Calibration** using reference images with known lifetimes
- **Component analysis** for multi-component systems
- **Phasor Mapping** — colormap apparent/normal lifetime, phasor phase, and phasor modulation per pixel, with an interactive 1D histogram and statistics table
- **FRET analysis** with donor trajectory visualization and multi-layer donor/background source selection
- **Filtering** with median, wavelet, and automatic thresholding (Otsu, Li, Yen)
- **Selections** via manual drawing, circular cursors, and automatic clustering
- **Exporting** results as OME-TIF or CSV (multiple layers simultaneously)

## Quick start

```bash
pip install napari-phasors
```

Then open napari and find the plugin under **Plugins → napari-phasors**.

For detailed installation instructions, see {doc}`installation`.

## How to use this documentation

**{doc}`Getting Started <installation>`** — Installation and sample data.

**{doc}`User Guide <guides/flim_workflow>`** — Step-by-step workflows for FLIM, hyperspectral analysis, calibration, filtering, selections, FRET, and exporting.

**{doc}`API Reference <api/index>`** — Programmatic API for all public modules.

## Citation

If napari-phasors contributes to a project that leads to a publication, please cite:

> doi: [10.5281/zenodo.14647626](https://doi.org/10.5281/zenodo.14647626)

```{tableofcontents}
```
