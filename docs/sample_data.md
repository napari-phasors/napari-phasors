# Sample Data

Two sample datasets for FLIM are provided, along with their corresponding calibration images. Additionally, a paramecium image is included as sample data for hyperspectral analysis.

To load sample data, go to **File → Open Sample → napari-phasors** and choose one of the available datasets.

![sample_data](https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/samples.gif)

## Available datasets

| Dataset | Type | Images included | Description |
|---------|------|----------------|-------------|
| Convallaria FLIM | FLIM | Image + calibration | Convallaria FLIM image and Rhodamine110 calibration (FBD format) |
| Embryo FLIM | FLIM | Image + calibration | FLUTE Embryo FLIM image and Fluorescein calibration (TIFF format) |
| Paramecium | Hyperspectral | Image | Paramecium hyperspectral image (LSM format) |

**Note:** The calibration images use reference solutions with known lifetimes:
- **Rhodamine110** (Convallaria calibration): 4 ns
- **Fluorescein** (Embryo calibration): 4 ns
