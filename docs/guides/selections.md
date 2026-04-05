# Selections


The **Selection** tab provides several modes for identifying regions of interest (ROIs) in the phasor plot and mapping the corresponding pixels back to the intensity image. The main selection modes are:

- **Cursor selection** (circular, polar, elliptical)
- **Automatic clustering** (k-means, GMM)
- **Manual selection** (freeform shapes)

Below, each mode is described in detail.

---

## Cursor selection

Define one or more cursors on the phasor plot to select regions of interest. Each cursor can be:

- Positioned by clicking and dragging on the plot (only circular and elliptical cursors)
- Switched between **Circular**, **Polar (Sector)**, and **Elliptical** modes
- Configured with specific coordinates and other parameters in the table

A separate labels layer is created for each cursor, color-coded for easy identification. Statistics for each cursor region are displayed in the table.


### Circular cursors

**Circular cursors** select a circular region in phasor space.

![circular_cursors](https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/circular%20cursors.gif)

### Polar cursors

**Polar cursors** allow selecting phasors within a range of phases and modulations.

![polar_cursors](https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/polar%20cursors.gif)

### Elliptical cursors

**Elliptical cursors** allow defining arbitrary stretching and rotation for specific phasor component analysis.

> [!TIP]
> To rotate (shift the angle of) an elliptical cursor, hold the **Shift** key and click-and-drag on the cursor.

![elliptical_cursors](https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/elliptical%20cursors.gif)

---

## Automatic clustering

Cluster phasors automatically using k-means or Gaussian Mixture Models (GMM). The number of clusters can be specified, and the results are mapped back to the image as a labels layer.

![clustering](https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/clustering.gif)

---

## Manual selection

Draw freeform shapes directly on the phasor plot using napari's built-in shapes tools. Pixels whose phasors fall inside the drawn shape are highlighted in the image.

![manual_selection](https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/manual%20selection.gif)
