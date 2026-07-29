# Time-lapse Analysis

Many acquisitions have one or more dimensions beyond the image plane — a time
series, a Z stack, or a spectral series read from a `.czi`, `.lsm` or `.ptu`
file. napari-phasors analyses every one of those planes, so the derived layers
(lifetime, component fractions, FRET efficiency) already play as a movie in the
napari viewer.

The **Frames** controls, shown beneath the phasor plot, extend that to the plot
itself: instead of pooling every timepoint into one cloud, the phasor plot, the
1-D histogram and the statistics table follow the timepoint napari is
displaying — including while napari plays the stack — and the result can be
exported as an animation.

The controls only appear when the selected layers actually have a non-spatial
axis. For plain 2-D data nothing changes.

## Frames controls

| Control | Description |
|---------|-------------|
| **Frames** | `All timepoints` pools the whole acquisition into one phasor plot (the default, and how earlier versions always behaved). `Current timepoint` shows only the frame displayed in the viewer. |
| **Axis** | Which dimension to step through. Only shown when the data has more than one non-spatial axis (e.g. a Z stack acquired over time). |
| **Export Animation…** | Render the time-lapse to an animated GIF. Enabled in `Current timepoint` mode. |

### Navigation is napari's slider

There is deliberately no frame slider, play button or frame counter in the
plugin: **napari's own dimension slider is the control**. Drag it, use its play
button, or type a frame number, and the phasor plot, the histogram and the
statistics table follow. The plugin also pushes changes back the other way, so
the image and the plot can never disagree about which timepoint you are
looking at.

The display mode and the chosen axis are stored in the layer's settings
alongside every other plot setting, so they are restored when you switch back
to a layer or reload an exported OME-TIF. The frame index itself is not stored
— it always comes from the viewer.

## What follows the frame

In `Current timepoint` mode:

- the **phasor plot** (density, scatter and contour) shows only that frame's
  pixels;
- the **phasor centers** and their statistics summarise only that frame;
- the **1-D histogram** of the analysis tabs (phasor mapping, component
  analysis, FRET) summarises only that frame;
- the **statistics table** switches to one row per timepoint (see below).

## Statistics table

In `All timepoints` mode the table keeps its usual layout: one row per layer,
pooled over the whole acquisition.

In `Current timepoint` mode it gains a **Frame** column and lists every
timepoint at once, so you can read the trend down the column instead of
scrubbing. The row for the frame on screen is **bold and highlighted**, and it
follows along as you move the slider or play the acquisition — the table also
scrolls to keep it in view. With several layers selected there is one row per
frame per layer, grouped by frame.

The centre-of-mass column is binned over the whole acquisition (using the
histogram's bin count and range), so the value is comparable from frame to
frame rather than being rebinned on each frame's own extent.

Group statistics stay a pooled, per-layer concept and are hidden while the
per-frame rows are shown.

Range sliders keep using the whole acquisition's extent, so the colormap
contrast limits do not jump around while playing. If a frame has no valid
pixels — because a threshold or a mask removed them all — the plot is blanked
rather than left showing the previous frame.

### A colour scale that means the same thing on every frame

The 2-D phasor histogram would otherwise re-bin and re-normalise on each
frame, so a given colour — and the colorbar beside it — would stand for a
different number of pixels at every timepoint. Instead, both the bin grid and
the colour range are computed once over the **whole acquisition**, across
**every selected layer**, and then applied unchanged to each frame:

- the bin grid is the one the `All timepoints` plot would draw, so the
  histogram does not shift under you while stepping;
- the colour scale runs from 1 count up to the highest bin count reached by
  any single frame, so the busiest frame uses the full colormap and quieter
  frames read as genuinely quieter.

Both are cached and only recomputed when the selected layers, the harmonic or
the bin count change.

### Selections

Selections are regions in phasor space, so they always apply to the whole
acquisition. Drawing a lasso, rectangle or ellipse while a single frame is
displayed labels the matching pixels in *every* timepoint, exactly as the
cursor and clustering selections do. This keeps a selection meaningful as a
population of molecular species rather than as a per-frame annotation.

## Exporting an animation

With `Current timepoint` active, click **Export Animation…** to choose:

- whether to render the **phasor plot**, the **histogram**, or both side by
  side;
- the **first** and **last** frame;
- the **frame rate**.

The result is written as an animated GIF. Exporting leaves the viewer on the
frame you started from.

```{note}
GIF export uses the `imageio` package, which ships with napari. If it is
missing from your environment, install it with `pip install imageio`.
```

## Exporting statistics

Both statistics tables offer a pooled and a per-timepoint export for stacks.

**Analysis statistics** — click **Export Table as CSV** in the Statistics dock
and choose:

- *Current view* — the table exactly as displayed;
- *All timepoints pooled* — one row per layer over the whole acquisition;
- *Per timepoint* — one row per layer per frame.

The per-timepoint file has the columns `Frame, Name, Center of Mass, Mean,
Median, Std Dev`.

**Phasor centers** — with phasor centers enabled in the Plot Settings tab,
click **Export Centers as CSV** and choose *All timepoints pooled* or *Per
timepoint*. The file has the columns `Frame, Name, G (center), S (center),
Phase (deg), Modulation`, which makes it straightforward to plot how the
phasor centroid moves over the acquisition.

## Example workflow

1. Open a time-lapse file and run the analysis you are interested in
   (see {doc}`phasor_mapping`, {doc}`component_analysis` or
   {doc}`fret_analysis`).
2. Inspect the pooled phasor plot to set thresholds, filters and selections.
3. Switch **Frames** to `Current timepoint`, then scrub or play napari's
   dimension slider to see how the distribution evolves.
4. Export the animation for a presentation, and the per-timepoint CSV for
   quantification.
