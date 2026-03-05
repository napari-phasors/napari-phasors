# Hyperspectral Phasor Workflow

This guide covers phasor analysis of hyperspectral imaging data.

## Loading hyperspectral data

Hyperspectral data can be loaded from supported file formats or the built-in paramecium sample dataset (**File → Open Sample → napari-phasors → Paramecium**).

## Plotting

Hyperspectral phasor data is plotted in the same **Phasor Plot** widget. To visualize data in the full universal circle, uncheck **Universal Semi-Circle/Full Polar Plot** in the **Plot Settings** tab.

![phasors_hyperspectral](https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/hsi%20plot.gif)

## Analysis

The same analysis tools (filtering, selections, component analysis) apply to hyperspectral data. The key difference is the interpretation:

- **FLIM phasors**: positions relate to fluorescence lifetimes
- **Hyperspectral phasors**: positions relate to spectral composition

All filtering, selection, and export workflows described in the FLIM guide apply identically to hyperspectral data.
