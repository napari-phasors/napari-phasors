(intro)=

# napari-phasors

napari-phasors is a comprehensive plugin, based on the [phasorpy](https://www.phasorpy.org/) library, that provides a complete workflow for phasor analysis in [napari](https://napari.org). It includes widgets for:

- **Reading** a wide range of FLIM and hyperspectral file formats:
	- **Raw data formats:**
		- `.ptu`, `.fbd`, `.sdt`, `.lsm`, `.tif`, `.tiff`, `.czi`, `.flif`, `.bh`, `.b&h`, `.bhz`, `.lif`, `.bin`, `.json`
	- **Processed data formats:**
		- `.ome.tif`, `.ome.tiff`, `.r64`, `.ref`, `.ifli`, `.lif`, `.json`
- **Phasor analysis** on multiple layers simultaneously, including support for stacking multiple raw data files
- **Calibration** using reference images with known lifetimes
- **Component analysis** for multi-component systems
- **Phasor Mapping** — colormap apparent/normal lifetime, phasor phase, and phasor modulation per pixel, with interactive 1D histograms, statistics tables, and arc overlay tools
- **FRET analysis** with donor trajectory visualization and multi-layer donor/background source selection
- **Filtering** with median, wavelet, and automatic thresholding (Otsu, Li, Yen)
- **Selections** via manual drawing, circular/polar/elliptical cursors, and automatic clustering
- **Exporting** results as OME-TIF or CSV (multiple layers simultaneously)

## Quick start

```bash
pip install napari-phasors
```

Then open napari and find the plugin under **Plugins → napari-phasors**.

For detailed installation instructions, see {doc}`installation`.

## How to use this documentation

**{doc}`Getting Started <installation>`** — Installation and sample data.

**{doc}`User Guide <guides/flim_workflow>`** — Step-by-step workflows for custom import, FLIM, hyperspectral analysis, calibration, filtering, selections, component analysis, phasor mapping, histogram/statistics, FRET, masking, and exporting.

**{doc}`API Reference <api/index>`** — Programmatic API for all public modules.

## Citation

If napari-phasors contributes to a project that leads to a publication, please cite:

> doi: [10.5281/zenodo.14647626](https://doi.org/10.5281/zenodo.14647626)

```{tableofcontents}
```
