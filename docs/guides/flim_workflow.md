# FLIM Phasor Workflow

This guide walks through a complete FLIM (Fluorescence Lifetime Imaging Microscopy) phasor analysis workflow using napari-phasors.

## 1. Load your FLIM data

napari-phasors can read several FLIM file formats directly:

- **FBD** files (`.fbd`) — FlimBox data
- **SDT** files (`.sdt`) — Becker & Hickl
- **PTU** files (`.ptu`) — PicoQuant
- **OME-TIF** files (`.ome.tif`) — Previously exported phasor data

Use **File → Open File(s)** or drag-and-drop your file into napari.

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

## 7. Export results

Export your phasor coordinates, selections, and analysis results as OME-TIF or CSV. See {doc}`exporting` for details.
