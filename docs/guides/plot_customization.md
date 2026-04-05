# Plot Customization and Examples

This page collects practical examples and visualization tools used with the
**Phasor Plot** and **Phasor Mapping** workflows.

## Example workflow (FLIM)

1. Load your FLIM file and open the **Phasor Plot** widget.
2. Calibrate your data (optional but recommended).
3. Go to the **Phasor Mapping** tab.
4. Choose **Lifetime** -> **Apparent Phase Lifetime** and enter the laser
   frequency.
5. Click **Calculate Output** to generate a false-color lifetime image.
6. Adjust display range and histogram settings as needed.
7. Export images and statistics for downstream analysis.

The GIF below shows the **Lifetime** workflow.

![lifetimes](https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/lifetime.gif)

The GIF below shows the **Phase** and **Modulation** color-mapping workflow.

![phase_modulation](https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/phase%20modulation.gif)

## Phasor plot customization

The **Plot Settings** tab offers additional ways to visualize phasor
distributions beyond the default 2D density histogram.

### Contour plot

Toggle **Contour Plot** to visualize density levels as smooth isolines.
This is useful when comparing overlapping layer distributions.

![contour_plot](https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/contour%20plot.gif)

### Phasor centers

Enable **Plot Centers** to overlay the center-of-mass coordinate for each
selected layer. This helps track global shifts between conditions.

![phasor_centers](https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/phasor%20centers.gif)
