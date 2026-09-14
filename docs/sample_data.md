# Sample Data

Two sample datasets for FLIM are provided, along with their corresponding calibration images. A paramecium image is included as sample data for hyperspectral analysis, and a FLIM-FRET training dataset is available for FRET analysis.

To load sample data, go to **File → Open Sample → napari-phasors** and choose one of the available datasets.

<video width="100%" autoplay loop muted playsinline poster="https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/sample%20data.gif">
  <source src="https://github.com/napari-phasors/napari-phasors-data/raw/main/videos/sample%20data.mp4" type="video/mp4">
</video>

## Available datasets

| Dataset | Type | Images included | Description |
|---------|------|----------------|-------------|
| Convallaria FLIM | FLIM | Image + calibration | Convallaria FLIM image and Rhodamine110 calibration (FBD format) |
| Embryo FLIM | FLIM | Image + calibration | FLUTE Embryo FLIM image and Fluorescein calibration (TIFF format) |
| Paramecium | Hyperspectral | Image | Paramecium hyperspectral image (LSM format) |
| FLIM-FRET Training Dataset | FLIM (FRET) | 4 images | Live HeLa cells: donor only, background autofluorescence and two FRET constructs (calibrated OME-TIFF) |

**Note:** The calibration images use reference solutions with known lifetimes:
- **Rhodamine110** (Convallaria calibration): 4 ns
- **Fluorescein** (Embryo calibration): 4 ns

## FLIM-FRET training dataset

The FLIM-FRET training dataset is intended for practicing FRET quantification
with the phasor approach (see {doc}`guides/fret_analysis`). It contains four
calibrated OME-TIFF files of live HeLa cells transfected with constructs based
on a donor fluorophore of mono-exponential lifetime τ ≈ 4.0 ns:

| File | Content | Expected FRET efficiency |
|------|---------|--------------------------|
| `Donor_Only.ome.tif` | Unquenched donor | ~0 % |
| `Background_Autofluorescence.ome.tif` | Non-transfected cells, for background/autofluorescence correction | – |
| `FRET_Construct_1.ome.tif` | Tandem donor–acceptor construct, short linker | ~60 % |
| `FRET_Construct_2.ome.tif` | Tandem donor–acceptor construct, flexible linker | ~30 % |

The data were acquired on an ISS FLIMbox frequency-domain FLIM system, calibrated
against Coumarin 6 in ethanol (τ = 2.50 ns) and median filtered in
napari-phasors before export. Because the phasor coordinates, harmonics,
calibration parameters and processing settings are stored in the OME-XML
metadata, loading the files restores the original filtering and thresholding
settings, as well as the 50 MHz laser frequency, directly.

The dataset is hosted on Zenodo:
[10.5281/zenodo.22261325](https://doi.org/10.5281/zenodo.22261325) (CC BY 4.0).
