# Selections

The **Selection** tab offers three modes for identifying regions of interest in the phasor plot and mapping the corresponding pixels back to the intensity image.

## Circular cursor selection

Define one or more circular cursors on the phasor plot. Each cursor can be:

- Positioned by clicking and dragging on the plot
- Resized interactively
- Configured with specific coordinates in the table

A separate labels layer is created for each cursor, color-coded for easy identification. Statistics for each cursor region are displayed in the table.

![circular_cursors](https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/circular%20cursors.gif)

## Automatic clustering

Cluster phasors automatically using k-means or Gaussian Mixture Models (GMM). The number of clusters can be specified, and the results are mapped back to the image as a labels layer.

![clustering](https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/clustering.gif)

## Manual selection

Draw freeform shapes directly on the phasor plot using napari's built-in shapes tools. Pixels whose phasors fall inside the drawn shape are highlighted in the image.

![manual_selection](https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/manual%20selection.gif)

## Masks

You can also create a mask in image space using a shapes or labels layer, and apply it to restrict the phasor plot to only those pixels.

![mask](https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/mask.gif)
