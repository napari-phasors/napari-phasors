# Component Analysis

The **Components** tab of the **Phasor Plot** widget lets you decompose phasor
distributions into fluorophore components and compute per-pixel fraction maps.
Two analysis modes are available:

- **Two-component linear projection**
- **Multi-component fit**

## Two-component linear projection

Use this mode when you want to project pixels onto a line between two
components in phasor space.

### Workflow

1. Open the **Components** tab and keep **Analysis Type** set to
   **Linear Projection**.
2. Define two component positions in the phasor plot (manually or with
   **Select**).
3. Click **Display Component Fraction Images**.
4. A fraction image is generated for the selected component(s), where each
   pixel value indicates its relative contribution.

This mode is fast and intuitive for mixtures dominated by two endmembers.

## Multi-component fit

Use this mode when your data contains more than two components.

### Workflow

1. Set **Analysis Type** to **Component Fit**.
2. Add and position the required number of components.
3. For higher component counts, define component positions across the required
   harmonics.
4. Click **Run Multi-Component Analysis**.
5. One fraction map per component is generated.

This mode uses multi-component fitting in phasor space and is appropriate for
more complex mixtures.

## Visualization and quantitative analysis

Component fraction results can be inspected visually as image layers and also
analyzed quantitatively in the **Histogram and Statistics Table** widget.

- View fraction distributions as histograms
- Compare layers as merged, individual, or grouped data
- Export summary statistics to CSV

For full histogram/table options, see {doc}`histogram_statistics`.
