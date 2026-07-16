# Filtering and Thresholding

The **Filter** tab provides tools to reduce noise in phasor data and remove low-signal pixels before analysis.

## Filters

### Median filter

Applies a spatial median filter to the phasor coordinates (G and S channels), reducing noise while preserving edges. Generally, filtering 3 times with a
3x3 kernel size is enough.

<video width="100%" autoplay loop muted playsinline poster="https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/median%20filter.gif">
  <source src="https://github.com/napari-phasors/napari-phasors-data/raw/main/videos/median%20filter.mp4" type="video/mp4">
</video>

### Wavelet (binlet pawFLIM) filter

Applies wavelet (binlet pawFLIM) denoising to the phasor data. This requires that the selected harmonics are compatible (each must have a double or half counterpart, e.g., harmonics 1 and 2).

<video width="100%" autoplay loop muted playsinline poster="https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/wavelet%20filter.gif">
  <source src="https://github.com/napari-phasors/napari-phasors-data/raw/main/videos/wavelet%20filter.mp4" type="video/mp4">
</video>

## Thresholding

Thresholding sets a minimum **and** maximum intensity cutoff using a range slider (with paired min/max value fields alongside it). Pixels with intensity below the minimum or above the maximum are masked (set to NaN) and excluded from the phasor plot and analysis.

<video width="100%" autoplay loop muted playsinline poster="https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/threshold.gif">
  <source src="https://github.com/napari-phasors/napari-phasors-data/raw/main/videos/threshold.mp4" type="video/mp4">
</video>

### Automatic threshold methods

Automatic methods compute the lower cutoff; the upper cutoff can still be adjusted manually with the max value field or the upper slider handle.

| Method | Algorithm | Best for |
|--------|-----------|----------|
| **Otsu** | Maximizes between-class variance | Bimodal histograms with two clear peaks |
| **Li** | Minimizes cross-entropy iteratively | Overlapping foreground/background distributions |
| **Yen** | Maximizes entropic correlation | Images where signal is a minority of pixels |

### Manual threshold

Select **Manual** from the threshold dropdown and drag either handle of the range slider (or edit the min/max value fields directly) to set custom lower and upper cutoff values.

## How it works

1. The intensity histogram of all selected layers is displayed
2. When an automatic method is selected, the optimal lower threshold is computed and the slider's lower handle is updated
3. All phasor coordinates from pixels outside the selected [min, max] intensity range are masked
4. The phasor plot updates in real time

The threshold values are stored in the layer metadata and are preserved across sessions when exporting to OME-TIF.
