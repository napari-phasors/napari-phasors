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
| **Phase** | Computes the polar angle of the phasor (radians or degrees) |
| **Modulation** | Computes the polar modulus of the phasor (0 – 1) |

### Lifetime mode

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

### Phase and Modulation modes

These modes derive the polar coordinates of the phasor directly with no
frequency input required.  A **Colormap** drop-down lets you choose the
colormap for the output layer; the default colormap for Phase is *hsv* and
for Modulation is *viridis*.

An optional **Apply colormap to 2D Histogram** checkbox, when enabled,
colors the phasor 2D histogram according to the phase or modulation value of
each phasor point, giving a spatially consistent color encoding between the
phasor plot and the image.

<video width="100%" autoplay loop muted playsinline poster="https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/phase%20modulation.gif">
  <source src="https://github.com/napari-phasors/napari-phasors-data/raw/main/videos/phase%20modulation.mp4" type="video/mp4">
</video>

## Calculating the output

Click **Calculate Output** to compute the selected metric for all currently
selected layers. A new napari image layer is created (or updated if it already
exists) with the selected colormap applied. The colormapped image can be
exported as a PNG or OME-TIF file. For histogram and statistics options, see
{doc}`histogram_statistics`.

For practical FLIM examples and visualization/customization tools (contours,
and phasor centers), see {doc}`plot_customization`.

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
