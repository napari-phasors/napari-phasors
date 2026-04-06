# Phasor Plot Customization

The **Plot Settings** tab offers several ways to visualize phasor distributions
and customize the appearance of the phasor plot.

## Plot Settings

In the **Plot Settings** tab, you can adjust the following general options:

- **White Background**: Toggle between a dark (default) and a white background for the plot.
- **Plot Type**: Choose how the phasor data is displayed. Options include **Density Plot**, **Scatter Plot**, **Contour Plot**, or **None** (to hide the main plot).

### Density Plot

The **Density Plot** displays a 2D histogram of the phasor coordinates.
Customizable parameters include:

- **Colormap**: Choose a colormap for the density distribution or a solid color gradient.
- **Bins**: Adjust the number of bins for the 2D histogram.
- **Log Scale**: Toggle logarithmic scaling for the color mapping to highlight low-density areas.

<video width="100%" autoplay loop muted playsinline poster="https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/phasor%20plot%202dhist.gif">
  <source src="https://github.com/napari-phasors/napari-phasors-data/raw/main/videos/phasor%20plot%202dhist.mp4" type="video/mp4">
</video>

### Scatter Plot

The **Scatter Plot** renders individual phasor points.
Customizable parameters include:

- **Marker Size**: Adjust the size of the points.
- **Color**: Select a color for the markers.
- **Alpha**: Set the transparency level of the points.

<video width="100%" autoplay loop muted playsinline poster="https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/phasor%20plot%20scatter.gif">
  <source src="https://github.com/napari-phasors/napari-phasors-data/raw/main/videos/phasor%20plot%20scatter.mp4" type="video/mp4">
</video>

### Contour Plot

The **Contour Plot** visualizes density levels as smooth isolines.
Customizable parameters include:

- **Colormap**: Select a colormap or a solid color for the contours.
- **Bins**: Adjust the number of bins of the underlying 2d histogram used to calculate contours.
- **Levels**: Define the number of contour levels.
- **Linewidth**: Adjust the thickness of the contour lines.

When multiple layers are selected, you can choose between **Merged**, **Individual**, or **Group** modes, each with customizable colors.

<video width="100%" autoplay loop muted playsinline poster="https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/phasor%20plot%20contour.gif">
  <source src="https://github.com/napari-phasors/napari-phasors-data/raw/main/videos/phasor%20plot%20contour.mp4" type="video/mp4">
</video>

## Phasor Centers

Enable **Plot Centers** to overlay the center-of-mass coordinates (mean or median) for each selected layer.
Customizable parameters include:

- **Method**: Choose between **Mean** or **Median** to calculate the phasor center.
- **Marker Size**: Adjust the size of the center markers.
- **Color**: Select a color for the markers.
- **Alpha**: Set the transparency level.

When multiple layers are selected, you can choose between **Merged**, **Individual**, or **Group** modes to visualize centers accordingly.

> [!TIP]
> Selecting **None** in the **Plot Type** can hide the main plot (density, scatter, or contour) and only show the phasor centers.

<video width="100%" autoplay loop muted playsinline poster="https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/phasor%20center.gif">
  <source src="https://github.com/napari-phasors/napari-phasors-data/raw/main/videos/phasor%20center.mp4" type="video/mp4">
</video>
