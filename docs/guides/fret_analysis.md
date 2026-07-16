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

<video width="100%" autoplay loop muted playsinline poster="https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/fret.gif">
  <source src="https://github.com/napari-phasors/napari-phasors-data/raw/main/videos/fret.mp4" type="video/mp4">
</video>

## Model parameters

The donor trajectory is a model of the donor channel signal, and its shape
depends on the following parameters, in addition to the donor lifetime and
frequency described above:

| Parameter | Description |
|---|---|
| **Donor Background** | The weight of the background signal in the donor channel, relative to the signal of the donor without FRET, in range 0–1. A value of 1 means the background and the FRET-free donor signal contribute equally; 0 means there is no background contribution. Increasing it pulls the whole trajectory toward the background phasor position. |
| **Background position** | The phasor coordinate (G, S) of the pure background signal, entered manually or averaged from one or more selected background layers (see step 2 above). Combined with **Donor Background**, it anchors where the trajectory is pulled toward. |
| **Proportion fretting** | The fraction of donor molecules that actually participate in FRET, in range 0–1. A value of 1.0 means every donor molecule has a nearby acceptor and undergoes energy transfer; lower values model a mixed population where some donors have no acceptor and never leave their unquenched (0 % efficiency) position. This shortens the trajectory but does not change its path. |
| **Overlay colormap on donor trajectory** | When enabled, colors the trajectory line by FRET efficiency (0–100 %) instead of drawing it as a single flat color, making it easier to read off the efficiency at a glance. |

## Donor lifetime types

| Type | Description |
|------|-------------|
| Apparent Phase Lifetime | Lifetime derived from the phase of the phasor |
| Apparent Modulation Lifetime | Lifetime derived from the modulation of the phasor |
| Normal Lifetime | Standard lifetime calculation |

## Results

FRET efficiency values are computed per-pixel and can be visualized as a colormapped image layer. The results can also be explored quantitatively in the **Histogram and Statistics Table** widget, allowing you to analyze the distribution and summary statistics of FRET efficiency across your data. Results can be exported to CSV.
