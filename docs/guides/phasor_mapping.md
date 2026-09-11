# Lifetime, phase and modulation quantification

The **Phasor Mapping** tab of the **Phasor Plot** widget lets you compute and
visualize pixel-level physical quantities derived from the phasor coordinates:
fluorescence lifetimes, phasor phase (angle), and phasor modulation (radius).
Each output is displayed as a colormapped image layer. Quantitative
distribution and summary metrics can be explored in the
**Histogram and Statistics Table** widget.

## Selecting the parameter to analyze

Use the **Parameter to Analyze** drop-down at the top of the tab to choose one
of three output modes:

| Mode | Description |
|------|-------------|
| **Lifetime** | Computes apparent phase lifetime, apparent modulation lifetime, or normal lifetime (ns) |
| **Phase** | Computes the polar angle of the phasor (radians) |
| **Modulation** | Computes the polar modulus of the phasor (0 – 1) |

## Lifetime mode

When **Lifetime** is selected, a secondary drop-down lets you choose between:

- **Apparent Phase Lifetime** — derived from the phase angle of the phasor
  $(\tau_\phi = \tan(\phi) / (2\pi f))$
- **Apparent Modulation Lifetime** — derived from the modulation
  $(\tau_m = \sqrt{1/m^2 - 1} / (2\pi f))$
- **Normal Lifetime** — computed from the phasor distance to the universal
  semicircle

You also need to set the **Frequency (MHz)** used in the acquisition (copied
from the calibration tab when calibration is applied).

<video width="100%" autoplay loop muted playsinline poster="https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/lifetime.gif">
  <source src="https://github.com/napari-phasors/napari-phasors-data/raw/main/videos/lifetime.mp4" type="video/mp4">
</video>

## Phase and Modulation modes

These modes derive the polar coordinates of the phasor directly with no
frequency input required.  A **Colormap** drop-down lets you choose the
colormap for the output layer; the default colormap for Phase is *cool* and
for Modulation is *PiYG*.

An optional **Apply colormap to 2D Histogram** checkbox, when enabled,
colors the phasor 2D histogram according to the phase or modulation value of
each phasor point, giving a spatially consistent color encoding between the
phasor plot and the image.

<video width="100%" autoplay loop muted playsinline poster="https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/phase%20modulation.gif">
  <source src="https://github.com/napari-phasors/napari-phasors-data/raw/main/videos/phase%20modulation.mp4" type="video/mp4">
</video>

## Mesh overlay

The **Mesh overlay** section draws a colored mesh of the selected quantity
behind the phasor data. It works in every mode, so you can read off a value
anywhere on the plot and compare it with the image colors. Turn it on with
**Show mesh overlay**. The controls that follow are:

| Control | Purpose |
|---|---|
| **Transparency** | Transparency of the mesh. |
| **Clip mesh to semicircle** | Only shows the mesh inside the universal semicircle (semicircle plot geometry only). |
| **Show colorbar** | Adds a colorbar for the mesh next to the phasor plot. |
| **Phase range (rad)** / **Modulation range** | *Phase and Modulation modes.* Restrict the mesh to a band of phase angles and/or modulations. |
| **Lifetime range (ns)** | *Lifetime mode.* Restricts the mesh to a band of lifetimes of the selected lifetime type. |

**Auto** fits a range to the plotted data. Each range is stored with the
layer. In Lifetime mode every lifetime type keeps its own range.

In Lifetime mode the mesh follows the selected lifetime type and needs the
**Frequency (MHz)**. It uses the displayed harmonic, just like the output map.
Lines of constant lifetime are:

- **Apparent Phase Lifetime**: rays from the origin (0, 0). This is the phase
  mesh in nanoseconds, so a lifetime range draws a wedge.
- **Apparent Modulation Lifetime**: circles centered on the origin. This is
  the modulation mesh in nanoseconds, so a lifetime range draws a ring.
- **Normal Lifetime**: rays from the semicircle center (0.5, 0). Each ray
  meets the universal semicircle at the single-exponential lifetime it stands
  for, so a lifetime range draws a wedge centered at (0.5, 0).

## Filtering by lifetime, phase or modulation

The **Filter** section works the way the **Filter** tab's intensity threshold
does, but on the quantities computed here: pixels whose value falls outside a
range you choose are invalidated and disappear from the phasor plot, the output
maps, the histogram and the statistics table.

Click **+ Add filter**, below the list, to create a filter on a specific quantity:

| Control | Purpose |
|---|---|
| Check box | Switch the filter off without deleting it. The pixels it hid come straight back. |
| Quantity drop-down | The lifetime, phase or modulation the filter tests. Changing it restarts the range at that quantity's full data span. |
| **Keep** / **Exclude** | Keep only what is inside the range, or remove what is inside it. |
| Range slider and min/max boxes | The range itself, in the metric's own units. |
| **×** | Remove the filter for good. |

Filters **combine as conditions, not as steps**: a pixel is kept only when it
satisfies every enabled criterion, and each criterion is always evaluated
against the unfiltered data. The order you add filters in does not change the result.

## Arc Overlay Tool

The **Phase & Modulation Arcs** tool helps visualize analysis boundaries by
overlaying:

- **Phase arcs**: constant lifetime trajectories
- **Modulation arcs**: constant modulation circles

These guides help interpret whether a distribution shifts along a single
component or reflects mixed lifetimes.

<video width="100%" autoplay loop muted playsinline poster="https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/phase%20arc.gif">
  <source src="https://github.com/napari-phasors/napari-phasors-data/raw/main/videos/phase%20arc.mp4" type="video/mp4">
</video>

## Calculating the output

Click **Calculate Output** to compute the selected metric for all currently
selected layers. A new napari image layer is created (or updated if it
already exists) with the selected colormap applied.

> [!TIP]
> Enable the **Autoupdate** toggle below the button to recompute the output
> automatically. While it is on, the button is disabled and the analysis is
> re-run whenever the tab's own inputs change, the harmonic or the layer
> selection changes, or the **Filter** or **Calibration** tab rewrites the
> phasor data.

After the first successful calculation, changing **Parameter to Analyze** or
the lifetime type automatically recalculates the new output for the currently
selected layers. Before the first calculation, these controls only configure
the requested output and do not start processing.

The histogram and statistics table follow the layers checked in **Phasor
Layers**. Unchecking a source removes its output from the histogram and hides
its derived image layer without deleting it. Rechecking the source restores
the existing output. Range changes affect only outputs whose source layers are
currently checked.

- View value distributions as histograms and summary statistics in the **Histogram and Statistics Table** widget
- Compare layers as merged, individual, or grouped data
- Export the colormapped image as a PNG or OME-TIF file
- Export summary statistics to CSV

For full histogram/table options, see {doc}`histogram_statistics`. For
practical FLIM examples and visualization/customization tools (contours and
phasor centers), see {doc}`plot_customization`.
