# Exporting Results

napari-phasors provides several export options for saving your analysis results.

## OME-TIF export

The average intensity image and phasor coordinates can be exported as OME-TIF
files. These files are compatible with both napari-phasors and PhasorPy,
allowing you to reload the data with all analysis settings preserved.
Multiple layers can be selected and exported simultaneously in a single operation.

## CSV export

Phasor coordinates and selections can be exported as CSV files using the **Export Phasor** widget. Analysis results — such as lifetime, FRET efficiency, and component fractions — can also be exported to CSV.

## Image export

The colormapped image layer can be exported with or without its associated colorbar.

![export_phasors](https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/export.gif)

## Copy settings

You can copy plot settings and analysis parameters (calibration, frequency, filter settings, component locations) from another layer or from a previously exported OME-TIF file, streamlining multi-dataset workflows.

![copy_settings](https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/copy%20settings.gif)
