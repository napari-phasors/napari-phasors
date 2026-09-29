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
- Change the width, offset, transparency, and color of the line joining the components
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
Selecting **Select from layer(s) phasor center** opens a dialog where you can select one or more image layers with phasor data. The widget will calculate the pooled phasor G and S centers of those layers and use it as the component coordinate.

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

> [!TIP]
> Enable the **Autoupdate** toggle below the button to recompute the fractions
> automatically. While it is on, the button is disabled and the analysis is
> re-run whenever the tab's own inputs change, the harmonic or the layer
> selection changes, or the **Filter** or **Calibration** tab rewrites the
> phasor data.

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

## Filtering by Component Fraction

The **Fraction filters** section lets you exclude or isolate specific phasor coordinates based on component fractions. Coordinates that fail the filter criteria are invalidated and set to `NaN`.

Each filter carries:

| Control | Purpose |
|---|---|
| Checkbox | Toggles the filter on or off without resetting its range. Hidden phasor coordinates are instantly restored when toggled off. |
| Component name | Identifies the target component, styled in that component's designated color. |
| **Keep** / **Exclude** | Retains coordinates inside the range (**Keep**) or drops them (**Exclude**). |
| Range slider and min/max boxes | Sets the fraction range, bounded by the values actually measured. |

Filters always evaluate against **the fractions of the latest analysis run**. Moving a component, switching between **Linear Projection** and **Component Fit**, or changing the harmonic updates all filter targets on the next analysis run—either immediately with **Autoupdate** on, or when triggered manually. Until then, active filters continue testing against the previous dataset.

> [!IMPORTANT]
> Fraction filters belong to the **same stack** as the criteria in the **Phasor Mapping** and **FRET** tabs. A pixel is kept only when *every* enabled criterion across all tabs keeps it. Because each criterion evaluates independently against the unfiltered data, the order you add them in does not change the result.

<video width="100%" autoplay loop muted playsinline poster="https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/component%20filter.gif">
  <source src="https://github.com/napari-phasors/napari-phasors-data/raw/main/videos/component%20filter.mp4" type="video/mp4">
</video>

### Pixel counts and percentages in the filter

The **Statistics** dock reports these metrics directly. Once a filter is active, the columns reflect the configured range—for example, *Component 1 Pixels in 0.2 – 0.8* and *Component 1 % in 0.2 – 0.8*.

The reported percentage is relative to the pixels that were **still valid** when the analysis ran—after accounting for the intensity threshold, median filter, image mask, and any active criteria from the **Phasor Mapping** and **FRET** tabs.

### Labels layer of filtered pixels

Check **Labels layer** to create a napari labels layer using the components' designated colors. Two options are available:

- **One layer per component** — generates a separate labels layer for each component, showing only the pixels kept by that component's range. Use this to isolate where a single component resides, or to overlay several components for direct comparison.
- **Single layer, dominant component** — combines all components into a single labels layer. Every pixel that satisfies *every* active filter is assigned the label of the component that accounts for its highest fraction. With no filters enabled, this acts as a direct classification map of the entire image, with each pixel labeled by its dominant component.

### Component colors

Each component card carries a color swatch at the right of its **G** and **S** fields, showing the color that component is drawn in on the phasor plot. Click it to choose a different one. The choice applies at once to the component's dot on the phasor plot, to its filter card, to both labels layouts, to its label in the combined layer and to its curve in the fraction histogram, so a component reads the same everywhere, and it is saved with the layer's settings rather than picked again every session. Once the analysis has run, the pick also recolors the component's fraction layer from black to the chosen color; in a Linear Projection, whose one layer holds both components, it sets that component's end of the colormap.

The color and the fraction layer's colormap follow whichever was changed last: choosing another colormap for the fraction layer in napari's layer controls recolors the dot, the card and the histogram curve to the colormap's highest value, replacing the color picked on the card.


## Visualization and quantitative analysis

Component fraction results can be inspected visually as image layers and also
analyzed quantitatively in the **Histogram and Statistics Table** widget.

- View fraction distributions as histograms
- Compare layers as merged, individual, or grouped data
- Export summary statistics to CSV

The **Component** selector above the histogram and the statistics table is
checkable: check several components to draw their fraction distributions in
the same plot. In *Merged* mode each component keeps its own curve, pooling
its own layers, so components are never averaged together; *Individual layers*
splits them further, one curve per component and image. The fraction range
slider spans every checked component and layer, and each component's curve is
drawn in the colormap of its own fraction layer, following it live when that
colormap, its contrast limits or its gamma change. The exception is a Linear
Projection pair: the second component's colormap is the first one reversed, so
two mirrored gradients would only confuse — both components are drawn in solid
colours instead. Either way, **Curve colours** in the Histogram Settings dialog
switches between the layer colormap and solid colours and lets you pick a
colour per component. When distributions have very different
pixel counts, enable **Normalize to maximum** in the Histogram Settings dialog
so each curve reaches 1.

Component names can be edited at any time: renaming one updates its curve, its
statistics columns and its fraction layers for every analysed image, and keeps
it checked in the selector.

Grouping still works on the analysed phasor layers: the group rows in the
Histogram Settings dialog list those layers, never one entry per component, and
a layer's group applies to every component curve derived from it.

The fraction image layers follow that selection: checking a component shows its
fraction layers in the viewer and hides the ones whose component is unchecked,
so the image on screen always matches the distributions being plotted.

The histogram and statistics table always reflect the layers currently checked
in **Phasor Layers**. Checking or unchecking a layer updates them immediately,
and the fraction image layers of unchecked layers are hidden rather than
deleted, so re-checking a layer restores its results straight away.

For full histogram/table options, see {doc}`histogram_statistics`.
