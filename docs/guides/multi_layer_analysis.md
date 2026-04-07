# Multi-Image Analysis and Settings Management

This guide explains how to perform simultaneous analysis on multiple data layers
and how to import or copy settings between layers and files.

## Simultaneous Multi-Image Analysis

The **Image Layers** selection dropdown in the **Phasor Plot** widget enables
simultaneous analysis across multiple datasets. By checking multiple layers,
you can perform phasor calculations, apply filters, and visualize overlapping
distributions in the phasor plot in a single, unified workflow.

While multiple layers can be active for plotting and processing, one layer is
always designated as the **Primary Layer**. You can identify or change the
primary layer by clicking the **Set as primary** button next to any layer name
in the selection dropdown.

The Primary Layer acts as the "source of truth" for the user interface:

- **Settings Display**: The parameters shown in the **Calibration**, **Filter**,
  and **Analysis** tabs always reflect the metadata and current configuration
  of the Primary Layer.
- **Workflow Focus**: When you adjust a setting (such as changing the laser
  frequency), the UI uses the Primary Layer as the active reference for
  updates. Any batch analysis performed will then sync these parameters across
  all other selected layers.

This batch-processing capability ensures consistency across datasets and
significantly streamlines the workflow for experimental series.

> [!TIP]
> You can quickly select or deselect all layers using the **All** and **None**
> links next to the layer selection dropdown.

## Copying and Importing Settings

You can reuse analysis parameters and plot configurations from existing layers
or from previously exported files. This feature helps maintain identical
processing pipelines for different samples.

The controls for loading settings are located in the main widget under
**Load and Apply Settings from:**.

### From Existing Layers

To copy settings from one layer to another:

1. Select the destination layer(s) in the **Image Layers** combobox.
2. Click the **Layer** button.
3. Select the source layer from the dialog.
4. Choose the categories of settings you wish to copy:
    - **Frequency & Harmonic**
    - **Plot Settings** (background, type, colormap, log scale, etc.)
    - **Calibration Parameters**
    - **Filter & Threshold Settings**
    - **Phasor Mapping Parameters**
    - **FRET & Component Analysis Positions**

<video width="100%" autoplay loop muted playsinline poster="https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/copy%20from%20layer.gif">
  <source src="https://github.com/napari-phasors/napari-phasors-data/raw/main/videos/copy%20from%20layer.mp4" type="video/mp4">
</video>


### From OME-TIF Files

Metadata stored in OME-TIF files exported by `napari-phasors` can be re-imported
to apply the same settings to newly opened data.

1. Click the **OME-TIFF File** button.
2. Select the `.ome.tif` file containing the desired settings.
3. Choose the settings categories to import.

This is particularly useful when returning to an analysis session or when
sharing standardized processing parameters with collaborators. For more
information on saving these settings with your data, see the {doc}`exporting` guide.

<video width="100%" autoplay loop muted playsinline poster="https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/copy%20from%20ometif.gif">
  <source src="https://github.com/napari-phasors/napari-phasors-data/raw/main/videos/copy%20from%20ometif.mp4" type="video/mp4">
</video>
