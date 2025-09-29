# napari-phasors

[![License BSD-3](https://img.shields.io/pypi/l/napari-phasors.svg?color=green)](https://github.com/napari-phasors/napari-phasors/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-phasors.svg?color=green)](https://pypi.org/project/napari-phasors)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-phasors.svg?color=green)](https://python.org)
[![tests](https://github.com/napari-phasors/napari-phasors/workflows/tests/badge.svg)](https://github.com/napari-phasors/napari-phasors/actions)
[![codecov](https://codecov.io/gh/napari-phasors/napari-phasors/branch/main/graph/badge.svg)](https://codecov.io/gh/napari-phasors/napari-phasors)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-phasors)](https://napari-hub.org/plugins/napari-phasors)

A comprehensive plugin for phasor analysis in napari. Based on the 
[phasorpy](https://www.phasorpy.org/) library.

[Jump to Intallation](#installation)

----------------------------------

## Usage

napari-phasors is a comprehensive plugin that provides a complete workflow 
for phasor analysis in napari. It includes widgets for reading various FLIM 
and hyperspectral file formats, performing phasor analysis, calibration, 
component analysis, FRET analysis, filtering, manual selections, and 
exporting results.

### Sample Data

Two sample datasets for FLIM are provided, along with their corresponding 
calibration images. Additionally, a paramecium image is included as sample 
data for hyperspectral analysis.

![sample_data](https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/samples.gif)

### Phasor Analysis

#### Plot FLIM Data

FLIM phasor data can be plotted as a 2D histogram or scatter plot in the
"Phasor Plot" widget. The colormap, the number of bins and the scale of the 
colors can be customized.

![phasors_flim](https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/flim%20plot.gif)

#### Plot Hyperspectral Data

Hyperspectral phasor data can also be plotted as a 2D histogram or scatter 
plot and visualized in the full universal circle. The 
'Universal Semi-Circle/Full Polar Plot' in the "Plot Settings" tab must be
unchecked.

![phasors_hyperspectral](https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/hsi%20plot.gif)

### Phasor Calibration

FLIM images can be calibrated using a reference image acquired under the same 
experimental parameters in the "Calibration" tab of the "Phasor Plot" widget. 
This reference image should consist of a homogeneous solution of a fluorophore
with a known fluorescence lifetime and the laser frequency used in the 
experiment. This ensures accuracy and consistency in lifetime measurements.

![calibration](https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/calibration.gif)

### Filtering and Thresholding

Apply various filters and thresholds to your phasor data to enhance analysis 
quality in the "Filter/Threshold" tab on the "Phasor Plot" widget. You can 
filter phasor coordinates using the median or wavelet filter.

![filter_threshold](https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/filter%20threshold.gif)

### Manual Phasor Selections

Create manual selections on the phasor plot to identify specific regions of 
interest. These selections can be used to highlight corresponding pixels in 
the intensity image and perform targeted analysis. Shape of selection can be
chosen at the top of the phasor plot and the selection ID can be selected in
the "Selection" tab of the "Phasor Plot" widget.

![selections](https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/selections.gif)


### Component Analysis

Perform multi-component analysis to identify and separate different 
fluorescent species in your sample. This feature allows you to decompose 
complex phasor distributions into individual components with distinct 
lifetimes. Two component analysis can be done in the "Components" tab of the
"Phasor Plot" widget.

![components](https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/components.gif)

### Apparent or Normal Lifetime Analysis

A FLIM image can be colormapped according to the phase or modulation apparent 
lifetime, as well as the normal lifetime in the "Lifetime" tab of the "Phasor 
Plot" widget. A histogram is also created for visualization of the distribution
of apparent lifetimes of the FLIM image.

![lifetimes](https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/lifetime.gif)

### FRET Analysis

Analyze Förster Resonance Energy Transfer (FRET) trajectories and efficiencies
in the "FRET" tab of the "Phasor Plot" widget.

![fret](https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/fret.gif)


### Phasor Custom Import

Supported file formats (`.tif`, `.ptu`, `.sdt`, `.fbd`, `.lsm`, `.ome.tif`) 
can be read and transformed to the phasor space in the "Phasor Custom Import" widget.
Additional options, such as the harmonics, channels and frames, can be 
specified depending on the file format to be read.

![custom_import](https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/import.gif)

### Data Export

The average intensity image and the phasor coordinates can be exported as 
OME-TIF files that can be read by napari-phasors and PhasorPy. Alternatively, 
the phasor coordinates, as well as the selections can be exported 
as a CSV file. This can be done in the "Export Phasor" widget

![export_phasors](https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/export.gif)

## Installation

You can install `napari-phasors` via [pip]. Follow these steps from a 
terminal.

We recommend using `miniforge` whenever possible. Click 
[here](https://github.com/conda-forge/miniforge?tab=readme-ov-file#miniforge) 
to choose the right download option for your OS.
**If you do not use `miniforge`, but rather Anaconda or Miniconda, replace 
the `mamba` term whenever you see it below with `conda`.**

Create a conda environment with napari by typing :

    mamba create -n napari-phasors-env napari pyqt python=3.10
    
Activate the environment :

    mamba activate napari-phasors-env

Install `napari-phasors` via [pip] :

    pip install napari-phasors

Alternatively, install latest development version with :

    pip install git+https://github.com/napari-phasors/napari-phasors.git

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"napari-phasors" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed 
description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[file an issue]: https://github.com/napari-phasors/napari-phasors/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/