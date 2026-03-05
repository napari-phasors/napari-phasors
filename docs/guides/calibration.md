# Calibration

FLIM images can be calibrated using a reference image acquired under the same experimental parameters. This is done in the **Calibration** tab of the **Phasor Plot** widget.

## What is calibration?

Phasor coordinates computed from raw FLIM data are affected by the instrument response function (IRF). Calibration corrects for this by using a reference sample — a homogeneous solution of a fluorophore with a **known fluorescence lifetime** — measured under the same conditions.

## How to calibrate

1. Load both your sample image and the reference image
2. Open the **Phasor Plot** widget and go to the **Calibration** tab
3. Select the reference layer from the dropdown
4. Enter the known lifetime of the reference fluorophore (in nanoseconds)
5. Enter the laser frequency used in the experiment (in MHz)
6. Click **Calibrate**

![calibration](https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/calibration.gif)

## Notes

- The reference image must have been acquired with the same laser frequency, detector settings, and number of time bins as the sample
- Calibration parameters are stored in the layer metadata and preserved when exporting to OME-TIF
- You can uncalibrate a layer to return to the raw phasor coordinates
