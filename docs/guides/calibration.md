# Calibration

FLIM images can be calibrated using a reference image acquired under the same experimental parameters. This is done in the **Calibration** tab of the **Phasor Plot** widget.

## What is calibration?

Phasor coordinates computed from raw FLIM data are affected by the instrument response function (IRF). Calibration corrects for this by using a reference sample, a homogeneous solution of a fluorophore with a **known fluorescence lifetime**, measured under the same conditions.

## How to calibrate

1. Load both your sample image and the reference image
2. Open the **Phasor Plot** widget and go to the **Calibration** tab
3. Select the reference layer from the dropdown
4. Enter the laser frequency used in the experiment (in MHz)
5. Enter the known lifetime of the reference fluorophore (in nanoseconds)
6. Click **Calibrate**

<video width="100%" autoplay loop muted playsinline poster="https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/calibration.gif">
  <source src="https://github.com/napari-phasors/napari-phasors-data/raw/main/videos/calibration.mp4" type="video/mp4">
</video>

## Reference fluorophore shortcut

If you are using a commonly used fluorophore, you can use the
**Reference fluorophore** dropdown in the **Reference parameters** section
and pick your fluorophore (each entry alsoshows the solution it is prepared
in). The **Lifetime (ns)** field is filled in automatically with the known
value. You can still type a custom lifetime by hand at any time; doing so
clears the dropdown selection.

The same lifetime values are available programmatically through the API via
{func}`napari_phasors.reference_lifetimes`, which returns the fluorophore name,
lifetime (ns) and solvent for each reference. The values are taken from
[ISS — Lifetime Data of Selected Fluorophores](https://iss.com/resources#lifetime-data-of-selected-fluorophores).

## Notes

- The reference image must have been acquired with the same laser frequency, detector settings, and number of time bins as the sample
- Calibration parameters are stored in the layer metadata and preserved when exporting to OME-TIF
- You can uncalibrate a layer to return to the raw phasor coordinates
