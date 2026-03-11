# Phasor Mapping

The **Phasor Mapping** tab of the **Phasor Plot** widget lets you compute and
visualize pixel-level physical quantities derived from the phasor coordinates:
fluorescence lifetimes, phasor phase (angle), and phasor modulation (radius).
Each output is displayed as a colormapped image layer and accompanied by an
interactive **1D Histogram** panel and a **Statistics Table**.

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

### Phase and Modulation modes

These modes derive the polar coordinates of the phasor directly with no
frequency input required.  A **Colormap** drop-down lets you choose the
colormap for the output layer; the default colormap for Phase is *hsv* and
for Modulation is *viridis*.

An optional **Apply colormap to 2D Histogram** checkbox, when enabled,
colors the phasor 2D histogram according to the phase or modulation value of
each phasor point, giving a spatially consistent color encoding between the
phasor plot and the image.

## Calculating the output

Click **Calculate Output** to compute the selected metric for all currently
selected layers. A new napari image layer is created (or updated if it already
exists) with the selected colormap applied. The colormapped image can be
exported as a PNG or OME-TIF file.

## 1D Histogram

After calculating the output, a **1D Histogram** dock panel opens
automatically below the Phasor Plot widget. It shows the distribution of the
computed metric values (lifetime, phase, or modulation) across all valid pixels
of all selected layers.

### Histogram features

- **Range slider** — drag the handles to clip the display / contrast limits of
  the colormapped image in real time.
- **Histogram Settings** — opens a dialog to configure:

  | Option | Description |
  |--------|-------------|
  | Display mode | **Merged** (pooled), **Individual layers** (one curve per layer), or **Grouped** (custom layer groups) |
  | Show standard deviation | Shades the ±1 SD band around the merged / group curve |
  | Show line | Overlays a vertical line at the *Center of mass*, *Mean*, or *Median* |
  | Show legend | Toggles the curve legend |
  | White background | Switches to a white plot background for figures |
  | Smooth curves | Applies Gaussian smoothing to improve curve readability |
  | Layer / group colours | Picker for per-layer or per-group histogram colours |

- **Save Histogram as PNG** — exports the histogram at 300 DPI.

## Statistics Table

A **Statistics** dock panel, linked to the histogram, displays per-layer (and
optionally per-group) descriptive statistics:

| Column | Description |
|--------|-------------|
| Name | Layer or group name |
| Center of Mass | Histogram-weighted center |
| Mean | Arithmetic mean |
| Median | 50th percentile |
| Std Dev | Standard deviation |

Right-clicking on the table provides **Copy**, **Copy with Headers**, and
**Select All** options. The entire table can also be exported to CSV via the
**Export Table as CSV** button.

## Example workflow (FLIM)

1. Load your FLIM file and open the **Phasor Plot** widget.
2. Calibrate your data (optional but recommended).
3. Go to the **Phasor Mapping** tab.
4. Choose **Lifetime** → **Apparent Phase Lifetime** and enter the laser
   frequency.
5. Click **Calculate Output** — a false-colour lifetime image and a histogram
   appear.
6. Adjust the range slider to set the displayed lifetime range.
7. Open **Histogram Settings** and switch to *Individual layers* if you have
   multiple layers loaded to compare their distributions side-by-side.
8. Use **Export Table as CSV** to save the statistics for further analysis.

![lifetimes](https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/lifetime.gif)
