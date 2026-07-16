# ROI Selection in the Phasor Plot


The **Selection** tab provides several modes for identifying regions of interest (ROIs) in the phasor plot and mapping the corresponding pixels back to the intensity image. The main selection modes are:

- **Cursor selection** (circular, polar, elliptical)
- **Automatic clustering** (GMM)
- **Manual selection** (freeform shapes)

Below, each mode is described in detail.

---

## Cursor selection

Define one or more cursors on the phasor plot to select regions of interest. Each cursor can be:

- Positioned and reshaped directly on the plot: circular and elliptical cursors are repositioned by clicking and dragging their body; polar cursors have no single body to drag — instead, click near any of its four edges (the two phase bounds or the two modulation bounds) and drag to move that edge independently
- Switched between **Circular**, **Polar (Sector)**, and **Elliptical** modes
- Configured with specific coordinates and other parameters in the table

A separate labels layer is created for each cursor, color-coded for easy identification. Statistics for each cursor region are displayed in the table.


### Circular cursors

**Circular cursors** select a circular region in phasor space, defined by:

- **Center G, S** — the coordinate of the circle's center, set by dragging the cursor on the plot or typed directly
- **Radius** — the circle's radius, set with the **Radius** field in the table

<video width="100%" autoplay loop muted playsinline poster="https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/circular%20cursor.gif">
  <source src="https://github.com/napari-phasors/napari-phasors-data/raw/main/videos/circular%20cursor.mp4" type="video/mp4">
</video>

### Polar cursors

**Polar cursors** select an annular wedge of phasor space, bounded by:

- **Phase (min / max)** — the lower and upper angular bounds, in degrees
- **Modulation (min / max)** — the lower and upper radial bounds (0 to 1)

Each of these four bounds is one edge of the wedge, and can be dragged directly in the phasor plot: click closest to the edge you want to move (whichever phase or modulation bound is nearest to the click) and drag it to a new position. The bounds can also be set numerically in the table.

<video width="100%" autoplay loop muted playsinline poster="https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/polar%20cursor.gif">
  <source src="https://github.com/napari-phasors/napari-phasors-data/raw/main/videos/polar%20cursor.mp4" type="video/mp4">
</video>

### Elliptical cursors

**Elliptical cursors** allow defining arbitrary stretching and rotation for specific phasor component analysis, defined by:

- **Center G, S** — the coordinate of the ellipse's center, set by dragging the cursor on the plot or typed directly
- **Radius** — the major-axis radius
- **rₘ** — the minor-axis radius
- **Angle** — the rotation of the major axis, in degrees

> [!TIP]
> To rotate (shift the angle of) an elliptical cursor, hold the **Shift** key and click-and-drag on the cursor.

<video width="100%" autoplay loop muted playsinline poster="https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/elliptical%20cursor.gif">
  <source src="https://github.com/napari-phasors/napari-phasors-data/raw/main/videos/elliptical%20cursor.mp4" type="video/mp4">
</video>

---

## Automatic clustering

Cluster phasors automatically using a Gaussian Mixture Model (GMM). The number of clusters can be specified, and the results are mapped back to the image as a labels layer, one ellipse per cluster.

Each ellipse is centered on the fitted Gaussian's mean, with its major/minor radii scaled from the eigenvalues of the fitted covariance by a fixed scaling factor (sigma = 2, not user-adjustable in this tab). At that default scaling, each ellipse is a **~98.2% confidence ellipse** of its cluster's distribution — i.e., about 98.2% of the pixels belonging to that Gaussian component are expected to fall inside the drawn ellipse.

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
