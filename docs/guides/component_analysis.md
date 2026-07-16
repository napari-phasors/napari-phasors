# Component Analysis

The **Components** tab of the **Phasor Plot** widget lets you decompose phasor distributions into fluorophore components and compute per-pixel fraction maps.

**Component locations can be selected in several ways:**
- Dragging components directly in the phasor plot.
- Specifying coordinates (G and S) or FLIM lifetime values manually.
- Using the **Select** dropdown options next to each component:
  - **Select on plot**: Click manually on the phasor plot.
  - **Select from layer(s) phasor center**: Calculate the location from the center of selected layer(s).
  - **Select from cursor center**: Use the center of any active cursor.
  - **Auto intersect semicircle**: Place the component at the intersection of the universal semicircle and the line connecting the previous component to the phasor center.

**The line and component locations can be fully customized:**
- Change the width, offset, alpha (transparency), and color of the line joining the components
- Overlay a histogram of the first component's fraction along the line (two-component Linear Projection only)
- Style the text labels for each component (size, bold, italic, color)

Two analysis modes are available:

- **Two-component fit projection**
- **Multi-component fit**

## Selecting component locations

To set or modify a component's coordinates, click the **Select** button next to the component row in the Components list. A dropdown menu provides the following methods:

### Select on plot
Allows manual selection by clicking on any position within the phasor plot. You can cancel the selection process at any time by pressing the **Esc** key on your keyboard.

<video width="100%" autoplay loop muted playsinline poster="https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/component%20select%20plot.gif">
  <source src="https://github.com/napari-phasors/napari-phasors-data/raw/main/videos/component%20select%20plot.mp4" type="video/mp4">
</video>

### Select from cursor center
Hovering over this option opens a submenu listing all active cursors (Circular, Polar, Elliptical, and GMM Clusters). Selecting a cursor instantly snaps the component to that cursor's center of gravity.

<video width="100%" autoplay loop muted playsinline poster="https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/component%20from%20cursor.gif">
  <source src="https://github.com/napari-phasors/napari-phasors-data/raw/main/videos/component%20from%20cursor.mp4" type="video/mp4">
</video>

### Select from phasor center (layer selection)
Selecting **Select from layer(s) phasor center** opens a dialog where you can select one or more image layers with phasor data. The widget will calculate the pooled phasor center of gravity of those layers and use it as the component coordinate.

<video width="100%" autoplay loop muted playsinline poster="https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/component%20from%20layers.gif">
  <source src="https://github.com/napari-phasors/napari-phasors-data/raw/main/videos/component%20from%20layers.mp4" type="video/mp4">
</video>

### Auto intersect semicircle
For Component 2 and subsequent components, selecting **Auto intersect semicircle** automatically positions the component on the universal semicircle. The position is calculated at the intersection point of the universal semicircle and a line drawn from the previous component's coordinates through the phasor center of the active data.

<video width="100%" autoplay loop muted playsinline poster="https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/component%20intersect%20semicircle.gif">
  <source src="https://github.com/napari-phasors/napari-phasors-data/raw/main/videos/component%20intersect%20semicircle.mp4" type="video/mp4">
</video>

## Two-component linear projection

Use this mode when you want to project pixels onto a line between two
components in phasor space.


1. Open the **Components** tab and keep **Analysis Type** set to
   **Linear Projection**.
2. Define two component positions in the phasor plot (manually or with
   **Select**).
3. Click **Display Component Fraction Images**.
4. A fraction image is generated for the selected component(s), where each
   pixel value indicates its relative contribution.

This mode is fast and intuitive for mixtures dominated by two endmembers.

<video width="100%" autoplay loop muted playsinline poster="https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/component%20linear%20analysis.gif">
  <source src="https://github.com/napari-phasors/napari-phasors-data/raw/main/videos/component%20linear%20analysis.mp4" type="video/mp4">
</video>

### Fraction histogram overlay

For a two-component Linear Projection, click **Edit Line Layout...** to open
the line and histogram settings, then enable **Overlay fraction histogram**.
This draws a histogram of the first component's fraction directly on top of
the line joining the two components, colored with the line's colormap, so the
distribution of pixel fractions is visible alongside the line itself in the
phasor plot. The histogram's height, transparency, and offset from the line
can all be adjusted in the same dialog.

## Multi-component fit

Use this mode when your data contains more than two components.

1. Set **Analysis Type** to **Component Fit**.
2. Add and position the required number of components.
3. For higher component counts, define component positions across the required
   harmonics.
4. Click **Run Multi-Component Analysis**.
5. One fraction map per component is generated.

This mode uses multi-component fitting in phasor space and is appropriate for
more complex mixtures.

<video width="100%" autoplay loop muted playsinline poster="https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/component%20fit%20analysis.gif">
  <source src="https://github.com/napari-phasors/napari-phasors-data/raw/main/videos/component%20fit%20analysis.mp4" type="video/mp4">
</video>

## Visualization and quantitative analysis

Component fraction results can be inspected visually as image layers and also
analyzed quantitatively in the **Histogram and Statistics Table** widget.

- View fraction distributions as histograms
- Compare layers as merged, individual, or grouped data
- Export summary statistics to CSV

For full histogram/table options, see {doc}`histogram_statistics`.
