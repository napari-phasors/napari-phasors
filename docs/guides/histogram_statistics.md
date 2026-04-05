# Histogram and Statistics Table

The **1D Histogram** and **Statistics Table** panels in napari-phasors provide quantitative analysis for a variety of workflows, including:

- **Component analysis**
- **Phasor mapping** (lifetime, phase, modulation)
- **FRET analysis**

For component-analysis-specific setup and interpretation, see {doc}`components`.

The statistics table can also display phasor center data for each layer or group.

## 1D Histogram

After running an analysis (component, mapping, or FRET), a **1D Histogram** dock panel opens automatically below the Phasor Plot widget. It shows the distribution of the computed metric values (e.g., lifetime, phase, modulation, or component fraction) across all valid pixels of all selected layers.

### Histogram display modes

The histogram can be shown in three modes:

- **Merged**: All selected layers' data are pooled and shown as a single curve.
- **Individual layers**: Each selected layer is shown as a separate curve, allowing direct comparison.
- **Grouped**: Layers can be assigned to custom groups, and each group's data is pooled and shown as a separate curve.

You can switch between these modes in the **Histogram Settings** dialog.

### Histogram features

- **Range slider**: Drag the handles to clip the display/contrast limits of the colormapped image in real time.
- **Show standard deviation**: Shades the ±1 SD band around the merged or group curve.
- **Show line**: Overlays a vertical line at the *Center of mass*, *Mean*, or *Median*.
- **Show legend**: Toggles the curve legend.
- **White background**: Switches to a white plot background for figures.
- **Smooth curves**: Applies Gaussian smoothing to improve curve readability.
- **Layer/group colours**: Picker for per-layer or per-group histogram colours.
- **Save Histogram as PNG**: Exports the histogram at 300 DPI.

## Statistics Table

The **Statistics** dock panel, linked to the histogram, displays per-layer (and optionally per-group) descriptive statistics for the currently displayed metric. The table can show statistics for:

- Lifetime, phase, or modulation (phasor mapping)
- Component fractions (component analysis)
- FRET efficiency or related metrics (FRET analysis)
- Phasor center coordinates (if enabled)

### Table columns

| Column | Description |
|--------|-------------|
| Name | Layer or group name |
| Center of Mass | Histogram-weighted center |
| Mean | Arithmetic mean |
| Median | 50th percentile |
| Std Dev | Standard deviation |
| (Phasor Center) | Center-of-mass coordinate in phasor space (if shown) |

Right-clicking on the table provides **Copy**, **Copy with Headers**, and **Select All** options. The entire table can also be exported to CSV via the **Export Table as CSV** button.

## Example usage

1. Run any supported analysis (component, mapping, or FRET).
2. View the histogram and statistics table for all selected layers.
3. Switch between merged, individual, or grouped display modes as needed.
4. Export the histogram or statistics for publication or further analysis.
