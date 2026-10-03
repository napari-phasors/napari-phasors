# Exporting Results

napari-phasors provides several export options for saving your analysis results.

In the **Export Phasor** widget, select the layers and pick the **File format**.
It defaults to OME-TIFF when a selected layer has phasor coordinates, to SVG
when only Labels layers are selected, and to PNG for other images. Only the
options that apply to the chosen format and layers are shown: the colorbar for
images, the colored PNG/TIFF option for Labels, the masked export for
OME-TIFF, and the **DPI** and **Figure height** (in inches, the width follows
the image's aspect ratio) when a figure is drawn.

## OME-TIF export

The average intensity image and phasor coordinates can be exported as OME-TIF
files. These files are compatible with both napari-phasors and PhasorPy,
allowing you to reload the data with all analysis settings preserved.
Multiple layers can be selected and exported simultaneously in a single operation.

A mask applied to the layer is saved in the file too, with its **Invert** flag
and label selection. When the file is opened again and the layer is selected
in the **Phasor Plot** widget, the mask comes back as a
`Restored Mask: <layer name>` layer and is applied again. A Shapes mask comes
back as an editable Shapes layer. The mask is stored compressed as an extra
page outside the phasor series, so it adds little to the file size and PhasorPy
still reads the file as before. **Export masked OME-TIFF** additionally sets
the pixels outside the mask to NaN in the exported phasor data.

## CSV export

Phasor coordinates and selections can be exported as CSV files using the **Export Phasor** widget. Analysis results, such as lifetime, FRET efficiency, and component fractions, can also be exported to CSV. Similarly, labels layers can be exported to CSV, where the pixel coordinates and corresponding label values/IDs are saved.

## Animation and per-timepoint export

For time-lapse acquisitions, the phasor plot and the histogram can be exported
as an animated GIF, and both statistics tables can be exported with one row per
timepoint. See {doc}`timelapse`.

## Image export

The colormapped image layer can be exported with or without its associated colorbar, at the chosen DPI and figure height. Labels layers can also be exported as images using their colored representation.

### Exporting and re-opening masks

A Labels layer exported as **PNG** or **TIFF** stores its label values, not a
colored picture, at the layer's own size. Open the file again (with napari's
own reader) and use **Convert to Labels** on the image layer to get the same
mask back. TIFF keeps any integer label value; PNG is 16-bit and holds labels up
to 65535, so export a layer with larger labels as TIFF. **JPEG** is always a
colored picture. To get a
colored PNG or TIFF of a Labels layer instead (for a figure), check **Export
Labels as colored PNG/TIFF** in the **Export Phasor** widget.

A Labels layer exported as **SVG** is drawn in color and also carries its exact
label values, scale and units. A Shapes layer exported as **SVG** keeps its
shapes and scale. Open either SVG file (**File > Open File(s)** or drag it into
napari) and the Labels or Shapes layer is restored, ready to be used as a mask
again. SVG files exported by earlier versions only hold the colored picture and
cannot be opened this way; export the layer again.

Shapes layers lose their units in SVG (the format has no place for them), so a
restored Shapes layer is in pixels; Labels layers keep their units.

<video width="100%" autoplay loop muted playsinline poster="https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/export.gif">
  <source src="https://github.com/napari-phasors/napari-phasors-data/raw/main/videos/export.mp4" type="video/mp4">
</video>
