# ROI Selection in the Phasor Plot


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

<video width="100%" autoplay loop muted playsinline poster="https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/circular%20cursor.gif">
  <source src="https://github.com/napari-phasors/napari-phasors-data/raw/main/videos/circular%20cursor.mp4" type="video/mp4">
</video>

### Polar cursors

**Polar cursors** allow selecting phasors within a range of phases and modulations.

<video width="100%" autoplay loop muted playsinline poster="https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/polar%20cursor.gif">
  <source src="https://github.com/napari-phasors/napari-phasors-data/raw/main/videos/polar%20cursor.mp4" type="video/mp4">
</video>

### Elliptical cursors

**Elliptical cursors** allow defining arbitrary stretching and rotation for specific phasor component analysis.

> [!TIP]
> To rotate (shift the angle of) an elliptical cursor, hold the **Shift** key and click-and-drag on the cursor.

<video width="100%" autoplay loop muted playsinline poster="https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/elliptical%20cursor.gif">
  <source src="https://github.com/napari-phasors/napari-phasors-data/raw/main/videos/elliptical%20cursor.mp4" type="video/mp4">
</video>

---

## Automatic clustering

Cluster phasors automatically using k-means or Gaussian Mixture Models (GMM). The number of clusters can be specified, and the results are mapped back to the image as a labels layer.

<video width="100%" autoplay loop muted playsinline poster="https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/automatic%20clustering.gif">
  <source src="https://github.com/napari-phasors/napari-phasors-data/raw/main/videos/automatic%20clustering.mp4" type="video/mp4">
</video>

---

## Manual selection


Use the top toolbar tools (from matplotlib) to draw circular, square, or freehand ROIs directly on the phasor plot. To select different regions, change the class (which also changes the color). To remove a class assigned in the phasor plot, assign class 0 to that region.

For circular and square selectors, once the shape is drawn, right-click to accept and apply the selection. You can use different manual selection IDs to compare multiple selections. Each selection creates a labels layer with the selected regions.

<video width="100%" autoplay loop muted playsinline poster="https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/manual%20selections.gif">
  <source src="https://github.com/napari-phasors/napari-phasors-data/raw/main/videos/manual%20selections.mp4" type="video/mp4">
</video>
