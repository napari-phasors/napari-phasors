# FRET Analysis

The **FRET** tab enables Förster Resonance Energy Transfer analysis using the phasor approach.

## Overview

FRET analysis in phasor space uses the donor fluorophore's position on the phasor plot and the trajectory it follows as energy transfer efficiency increases. By measuring where a sample falls along this trajectory, the FRET efficiency can be determined.

## Workflow

1. **Set the donor lifetime**: Enter it manually, or select one or more donor
   layers from the **Donor Source** drop-down. If multiple layers are selected, the donor lifetime is computed automatically from the average phasor position of all selected layers.
2. **Set the background position** (optional): Select one or more background layers from the **Background Source** drop-down, or enter coordinates manually to correct for autofluorescence. If multiple layers are selected, the background position is computed as the average of the selected layers.
3. **Configure the frequency**: The laser frequency used in the experiment.
4. **Visualize the trajectory**: The donor trajectory is drawn on the phasor
   plot showing the path from 0 % to 100 % FRET efficiency.

## Donor lifetime types

| Type | Description |
|------|-------------|
| Apparent Phase Lifetime | Lifetime derived from the phase of the phasor |
| Apparent Modulation Lifetime | Lifetime derived from the modulation of the phasor |
| Normal Lifetime | Standard lifetime calculation |

## Results

FRET efficiency values are computed per-pixel and can be visualized as a colormapped image layer. The results can also be explored quantitatively in the **Histogram and Statistics Table** widget, allowing you to analyze the distribution and summary statistics of FRET efficiency across your data. Results can be exported to CSV.
