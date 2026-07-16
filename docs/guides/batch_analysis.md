# Batch Analysis

The **Batch Analysis** widget applies the same reading and analysis pipeline
to every supported file in a folder, then exports the results, without
requiring you to open each file individually in the viewer.

Open it from:

**Plugins -> napari-phasors -> Batch Analysis**

## Overview

The widget is organized into tabs, one per pipeline stage:

- **Setup** — input folder, file format, read parameters, export destination
- **Signal Export** — export the average signal (decay/spectrum) per file
- **Phasor Plot Settings** — shared styling/export options for phasor plots
- **Calibration** — calibrate against a reference of known lifetime
- **Filter & Threshold** — denoise phasor coordinates and mask dim pixels
- **Masks** — restrict analysis to a region of interest per file
- **Selection** — manual cursors or automatic clustering
- **Components** — multi-component fraction analysis
- **Phasor Mapping** — lifetime, phase, and modulation images
- **FRET** — apparent FRET efficiency images

Only **Setup** is mandatory: it requires an **input folder** and an **export
folder**. Every other analysis tab (Signal Export, Calibration, Filter &
Threshold, Masks, Selection, Components, Phasor Mapping, FRET) has an
**Enable** toggle at the top; the rest of the tab's controls are greyed out
until it is switched on, and clicking a disabled control flashes a reminder
next to the toggle. Fields marked with a red <span style='color:#e74c3c;'>*</span>
are required for the tab to run.

Calibration and Filter & Threshold, if enabled, are applied first (in that
order) since they change the phasor coordinates that every other analysis
reads. Masks restrict which pixels are analyzed. The remaining analysis tabs
(Signal Export, Selection, Components, Phasor Mapping, FRET) are independent
of each other and can be enabled in any combination in a single run.

Once configured, click **Run batch analysis** at the bottom of the widget. A
progress bar and status label track the run across all files.

<video width="100%" autoplay loop muted playsinline poster="https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/batch%20analysis.gif">
  <source src="https://github.com/napari-phasors/napari-phasors-data/raw/main/videos/batch%20analysis.mp4" type="video/mp4">
</video>

## Setup

Configures which files are processed, how they are read, and where and how
results are exported.

### Input folder

| Parameter | Description |
|---|---|
| **Copy settings from layer / OME-TIFF…** | Populates the analysis tabs (calibration, filter, phasor mapping, FRET, and component positions) from the settings stored in an already-loaded phasor layer, or from an `.ome.tif`/`.ome.tiff` file written by napari-phasors. |
| **Folder containing the files to process** *(required)* | The folder scanned for input files. |
| **Include subfolders** | When checked, files are also discovered recursively in subfolders. |
| **File format** | The file extension to process from the scanned folder (e.g. `.ptu`, `.sdt`, `.ome.tif`). Only one format is processed per run. |

### Read parameters

| Parameter | Description |
|---|---|
| **Harmonics** | Comma-separated harmonics to read (e.g. `1, 2`), `all`, or empty for the reader's default. |
| **Reader keyword arguments** | Format-specific reader parameters (e.g. channel, frame) shown as typed fields for the selected format, plus a **+ Add reader keyword argument** button for any additional key/value pair to pass to the reader. |

### Export

| Parameter | Description |
|---|---|
| **Destination folder for the results** *(required)* | Where all exported files are written. |
| **Export Intensity Image (with phasor data) as** | One or more of **OME-TIFF** (preserves phasor data and settings, re-importable), **CSV**, and **Image (PNG)**. |
| **Preserve relative subfolder structure** | Mirrors the input folder's subfolder layout inside the export folder instead of flattening all outputs into one folder. |
| **Include colorbar in exported analysis images (PNG)** | Draws a colorbar next to colormapped analysis images (component fractions, phasor mapping, FRET). |
| **Image DPI** | Rendering resolution for exported PNG images, phasor plots, and histograms: Low (100), Mid (300), or High (600). Higher DPI gives sharper but larger/slower exports. |
| **Filename suffix** | Optional text appended to every exported filename. |
| **Also load results into the viewer** | In addition to exporting to disk, adds the resulting layers to the napari viewer. |

### Performance

| Parameter | Description |
|---|---|
| **Bounded memory (streaming aggregation)** | Accumulates fixed-bin histograms instead of holding every pixel in memory for grouped contour/histogram outputs. With this on, median and center-of-mass values become histogram approximations. Useful for very large batches. |
| **Parallel workers** | Number of files read and computed concurrently (1 = single-threaded). File writing and plot rendering always stay on the main thread. |

## Signal Export

Exports the average signal along the axis the phasor is computed on (time
bins for FLIM, wavelength for hyperspectral data). For raw files the signal is
averaged per pixel over the batch mask (see **Masks**); for processed
OME-TIFF files written by napari-phasors, the stored signal (summed over all
pixels at import) is used instead and the mask does not apply. This tab is
locked with an explanation if the selected format cannot provide a signal
(e.g. processed R64/REF/IFLI files).

| Parameter | Description |
|---|---|
| **Export individual signal plots (one per file)** | Exports one signal plot per file, drawn like the signal preview in the Custom Import widget. |
| **Line color** | Color of the individual signal line. |
| **Export combined signal plot (mean ± shaded SD)** | Overlays every file's signal as a per-group mean line with a shaded ±1 standard-deviation band. |
| **Configure Groups…** | Assigns files to groups for the combined plot and sets each group's color/legend. Grouping is shared with the **Phasor Plot Settings** tab. |
| **Export plots as** | Output formats for the signal plots: PNG (rendered plot) and/or CSV (underlying values). |
| **Normalization** | Scales each signal before plotting so files of different intensities are comparable: **None** (average per pixel), **Peak** (max = 1), or **Area** (sum = 1). |
| **Channels** | For multichannel files, draw each channel in its own plot (**Separate**) or overlay all channels in one plot (**Together**). Single-channel files are unaffected. |

## Phasor Plot Settings

Styles and exports the phasor plots shared by every analysis tab. Individual
(per-file) and combined (all files pooled) plots are configured
independently, and the chosen styles propagate to the analysis-tab overlay
plots (Components, Phasor Mapping, FRET, Selection).

### Common appearance

| Parameter | Description |
|---|---|
| **Semicircle (FLIM) / Full polar plot (HSI)** | Toggles between drawing the universal semicircle (single-exponential reference, for FLIM data) or the full polar plot (for hyperspectral data). |
| **White background** | Renders exported plots on a white background instead of the napari theme background. |
| **Export legends** | Includes a legend in exported phasor plots. |
| **Frequency (MHz)** | Frequency used to place lifetime tick marks on the semicircle of the exported plot. |
| **Show phasor centers / color** | Marks each file's phasor center (mean G/S) on the plot, in the chosen color. |
| **Export phasor centers (CSV, all harmonics)** | Writes a CSV with each file's phasor center for every harmonic. |

### Individual and combined plots

Both the **Individual phasor plots** and **Combined phasor plot** sections
share the same style controls:

| Parameter | Description |
|---|---|
| **Export individual/combined phasor plots** | Master checkbox for exporting this plot mode. |
| **Plot type** | **Histogram** (2-D density), **Scatter** (dot plot), **Contour** (density contour lines), or **None** (export no phasor data, e.g. only the phasor centers). |
| **Colormap** | Colormap for the density (Histogram/Contour). |
| **Histogram bins** | Number of bins per axis for the histogram/contour density. |
| **Log scale** | Uses a logarithmic color scale for the histogram density (Histogram only). |
| **Marker size / color / alpha** | Scatter-only: size, color, and opacity of the scatter markers. |
| **Contour levels / linewidth** | Contour-only: number of contour lines and their line width. |

The combined plot additionally has:

| Parameter | Description |
|---|---|
| **Combined contour mode** | **Merged** pools every file into a single contour; **Grouped** draws one contour per file group (Contour plot type only). |
| **Configure Groups…** | Assigns files to groups and sets group colors/legend for the grouped contour (shared with Signal Export). |

### Zoomed section

| Parameter | Description |
|---|---|
| **Export a zoomed section** | In addition to the full plot, saves a `_zoom` PNG cropped to the limits below, for both individual and combined plots and all analysis-tab overlay plots. |
| **G (x) range / S (y) range** | Zoom limits in phasor coordinates. |
| **Draw the zoom region as a rectangle on the full plot** | Outlines the zoomed region with a rectangle on the full (un-cropped) plot. |

## Calibration

Calibrates phasor coordinates against a reference of known lifetime before
the rest of the pipeline runs. See {doc}`calibration` for the concept.

| Parameter | Description |
|---|---|
| **Reference source** | Where the calibration reference comes from: **Same reference for all** files, **Different reference per subfolder**, or **Use copied phase/modulation** (from settings copied in the Setup tab). |
| **Reference layer** *(required, "Same" mode)* | A phasor layer already loaded in the viewer, used as the reference for every file. |
| **Reference file** *(alternative to Reference layer)* | Path to a reference file read directly instead of using a loaded layer. |
| **Per-subfolder reference files** *("Different reference per subfolder" mode)* | One reference file picker per subfolder discovered in the input folder. |
| **Frequency (MHz)** *(required)* | Laser repetition/modulation frequency at which the data and reference were acquired. |
| **Reference lifetime (ns)** *(required)* | Known fluorescence lifetime of the calibration reference. |

## Filter & Threshold

Smooths phasor coordinates and excludes low-intensity pixels before any
analysis runs. See {doc}`filtering_thresholding` for the concept.

### Filter

| Parameter | Description |
|---|---|
| **Filter method** | **None**, **Median**, or **Wavelet (binlet pawFLIM)**. |
| **Median kernel size** *(Median)* | Side length (pixels) of the square median-filter window; larger windows smooth more. |
| **Median repetitions** *(Median)* | Number of times the median filter is applied in succession. |
| **Wavelet sigma** *(Wavelet)* | Noise standard deviation used by the wavelet (binlet pawFLIM) filter; higher values remove more noise. Requires harmonics with a double/half counterpart (e.g. 1 and 2). |
| **Wavelet levels** *(Wavelet)* | Number of wavelet decomposition levels used for denoising. |

### Threshold

| Parameter | Description |
|---|---|
| **Threshold method** | **None**, **Manual**, or automatic (**Otsu**, **Li**, **Yen**). |
| **Threshold (min)** *(Manual)* | Minimum mean intensity a pixel must have to be kept; dimmer pixels are excluded (set to NaN). |
| **Threshold (max)** *(Manual)* | Maximum mean intensity a pixel may have to be kept; set to `none` to disable the upper bound. |

## Masks

Restricts each image to a region of interest using mask image files matched
to inputs by name (e.g. `ABC.ptu` ↔ `ABC_mask.png`). See {doc}`mask` for the
interactive-widget equivalent.

| Parameter | Description |
|---|---|
| **Add mask folder…** | Adds a folder to scan for mask image files (multiple folders can be added). |
| **Clear** | Removes all added mask folders. |
| **Include mask subfolders** | Scans mask folders recursively. |
| **Per-file mask row** | For each matched input file, a dropdown to pick its mask file and an **Invert** checkbox. Pixels outside the mask (or inside it, if inverted) are excluded (set to NaN) before analysis. |


## Selection

Labels phasor regions with manual cursors or automatic GMM clustering, and
exports a selection (labels) image per file. See {doc}`phasor_selection` for
the interactive-widget equivalent.

| Parameter | Description |
|---|---|
| **Harmonic** | Harmonic whose phasor coordinates the cursors/clustering act on. |
| **Mode** | **Manual cursors** or **Automatic clustering (GMM)**. |

### Manual cursors

Each cursor row (**+ Add cursor**) has:

| Parameter | Description |
|---|---|
| **Type** | **Circular**, **Elliptic**, or **Polar**. |
| **Color** | Display color of the cursor region. |
| **G, S** *(Circular, Elliptic)* | Center coordinates of the cursor. |
| **r** *(Circular, Elliptic)* | Radius (major radius for Elliptic). |
| **rₘ** *(Elliptic)* | Minor radius. |
| **∠** *(Elliptic)* | Rotation angle, in degrees. |
| **φ₋, φ₊** *(Polar)* | Minimum/maximum phase bounds, in radians. |
| **m₋, m₊** *(Polar)* | Minimum/maximum modulation bounds. |

### Automatic clustering (GMM)

| Parameter | Description |
|---|---|
| **Number of clusters** | Number of clusters (regions) the GMM fits to the phasor data. |
| **Sigma** | Size of each cluster's elliptic region, in standard deviations of the fitted Gaussian. |

### Outputs

| Parameter | Description |
|---|---|
| **Export cursor selections as** | PNG (rendered image) and/or CSV (label IDs). |
| **Export selection statistics (CSV)** | Writes one CSV with, per cursor/cluster, the pixel count inside the region and its percentage of the valid pixels. |
| **Export Phasor Plot with Selection Overlay (PNG)** | Exports the phasor plot with the cursor/cluster regions overlaid. |

## Components

Decomposes each pixel into fractional contributions of known phasor
components. See {doc}`component_analysis` for the interactive-widget
equivalent and for the analysis-type concepts.

| Parameter | Description |
|---|---|
| **Analysis type** | **Linear Projection** (two components) or **Component Fit** (multi-component fit). |
| **Frequency (MHz)** | Used to convert a typed lifetime into G/S coordinates. |
| **Harmonic** | Harmonic whose component G/S locations are currently being edited. Fits with more than 3 components need locations at several harmonics: switch this selector and enter G/S for each required harmonic. |
| **Component G/S locations** *(required)* | One row per component (**+ Add component**), each with a name, G, S, and an optional lifetime (ns) field that, when a value is entered, converts to G/S automatically. Each component also has its own fraction-image colormap. |

### Fraction images and phasor plot styling

| Parameter | Description |
|---|---|
| **Range (image & histogram)** | Fraction range for the colormapped fraction images and histogram. Uncheck **Auto** to set fixed min/max limits; **Auto** pools a single range across every file so outputs are comparable. |
| **Edit line layout…** | Style of the dots and connecting line drawn between components in the exported phasor plot. |
| **Edit component name layout…** | Style of the component name labels drawn in the exported phasor plot. |

### Outputs

Formats for exporting the component fraction images, per-file histogram,
statistics table (CSV), and the phasor plot with the components overlay. See
[Outputs (shared pattern)](#outputs-shared-pattern) below.


## Phasor Mapping

Maps each pixel's phasor to a physical quantity and exports it as an image
per file. See {doc}`phasor_mapping` for the interactive-widget equivalent.

| Parameter | Description |
|---|---|
| **Outputs** *(required)* | One or more of **Apparent Phase Lifetime**, **Apparent Modulation Lifetime**, **Normal Lifetime**, **Phase**, and **Modulation**. One image is exported per file per selected output. |
| **Frequency (MHz)** | Used to convert phasor coordinates to lifetimes; only enabled when a lifetime output is selected. |
| **Harmonic** | Harmonic used for the mapping. |
| **Colormap** | Colormap applied to the exported mapped images. |
| **Range (image & histogram)** | Value range (e.g. lifetime) for the colormapped image and histogram. Uncheck **Auto** to set fixed min/max limits; **Auto** pools a single range across every file. |

### Phasor plot mesh and coloring

| Parameter | Description |
|---|---|
| **Color phasor by** | Colors the exported phasor plot's points by **None**, **Phase**, or **Modulation**. |
| **Mesh overlay** | Draws a **Phase mesh** and/or **Modulation mesh** grid over the phasor plot. |
| **Mesh/color colormap** | Colormap used for the mesh and/or point coloring. |
| **Mesh alpha** | Opacity of the mesh overlay. |
| **Clip mesh to semicircle** | Only shows the mesh inside the universal semicircle (semicircle plot geometry only). |
| **Range** | **Auto (range from all files)** computes a single phase/modulation range pooled across every file at export; uncheck to set fixed ranges manually. |
| **Phase range (rad)** | Manual minimum/maximum phase for the mesh, when Auto is off. |
| **Modulation range** | Manual minimum/maximum modulation for the mesh, when Auto is off. |

### Outputs

Formats for exporting the mapped images, per-file histogram, statistics table
(CSV), and the phasor plot with the mesh/coloring overlay. See
[Outputs (shared pattern)](#outputs-shared-pattern) below.


## FRET

Estimates the apparent FRET efficiency of each pixel from a donor trajectory,
and exports it as an image per file. See {doc}`fret_analysis` for the
interactive-widget equivalent and the underlying concept.

### Donor trajectory

| Parameter | Description |
|---|---|
| **Donor lifetime (ns)** *(required)* | Fluorescence lifetime of the donor in the absence of FRET; anchors the donor trajectory. |
| **Frequency (MHz)** *(required)* | Laser repetition/modulation frequency at which the data were acquired. |
| **Harmonic** | Harmonic used to compute the FRET efficiency. |
| **Donor background** | Fraction of the donor signal coming from background; shifts the trajectory toward the background phasor. |
| **Donor fretting fraction** | Fraction of donor molecules that undergo FRET (1.0 = all donors participate). |
| **Background G, Background S** | G/S coordinates of the background phasor. |

### Image styling

| Parameter | Description |
|---|---|
| **Colormap** | Colormap applied to the exported FRET-efficiency images. |
| **Range (image & histogram)** | Efficiency range for the colormapped image and histogram. Uncheck **Auto** to set fixed min/max limits; **Auto** pools a single range across every file. |

### Outputs

Formats for exporting the FRET-efficiency images, per-file histogram,
statistics table (CSV), and the phasor plot with the FRET overlay. See
[Outputs (shared pattern)](#outputs-shared-pattern) below.


## Outputs (shared pattern)

The Components, Phasor Mapping, and FRET tabs share the same **Outputs**
section layout:

| Parameter | Description |
|---|---|
| **Export \<analysis\> as** | Formats for the colormapped analysis image (one per file): PNG (rendered image) and/or CSV (raw per-pixel values). |
| **Export histogram as** | Formats for the per-file histogram of the analysis values: PNG and/or CSV. |
| **Configure groups and display…** | Enabled once a histogram format is selected. Sets the merged/individual/grouped display mode (shared across tabs), assigns files to groups for combined grouped histograms/statistics, and sets histogram display options. |
| **Export statistics table (CSV)** | Writes per-file (and per-group, if grouped) descriptive statistics. See {doc}`histogram_statistics`. |
| **Export Phasor Plot with \<analysis\> Overlay (PNG)** | Exports the phasor plot styled per the **Phasor Plot Settings** tab, with the tab's analysis overlay (components, mesh, or FRET trajectory) drawn on top. |

## Running the batch

Click **Run batch analysis** to process every file of the selected format in
the input folder. A progress bar tracks per-file progress and the status
label reports completion or the first error encountered. The **Run batch
analysis** button stays disabled until the required Setup fields are filled
in and at least one export option (Setup export types, or an enabled
analysis tab's outputs) is selected.
