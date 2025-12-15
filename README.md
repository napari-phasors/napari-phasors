# napari-phasors

[![License BSD-3](https://img.shields.io/pypi/l/napari-phasors.svg?color=green)](https://github.com/napari-phasors/napari-phasors/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-phasors.svg?color=green)](https://pypi.org/project/napari-phasors)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-phasors.svg?color=green)](https://python.org)
[![tests](https://github.com/napari-phasors/napari-phasors/workflows/tests/badge.svg)](https://github.com/napari-phasors/napari-phasors/actions)
[![codecov](https://codecov.io/gh/napari-phasors/napari-phasors/branch/main/graph/badge.svg)](https://codecov.io/gh/napari-phasors/napari-phasors)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-phasors)](https://napari-hub.org/plugins/napari-phasors)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14647626.svg)](https://doi.org/10.5281/zenodo.14647626)

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

### Mask

![mask](https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/mask.gif)

You can create a mask using either a shapes layer or a labels layer in napari. Once the mask is created, select it from the mask combobox in the "Phasor Plot" widget. Only the pixels inside the selected mask will be plotted in the phasor space and included in subsequent analyses. This allows you to focus your analysis on specific regions of interest within your data.

### Copy Settings and Analysis

![copy_settings](https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/copy%20settings.gif)

You can quickly copy plot settings and analysis parameters, such as calibration, frequency, filter settings, and component locations, from another layer or from an OME-TIF file previously exported with the napari-phasors plugin. This feature streamlines the workflow by allowing you to reuse established configurations, ensuring consistency and saving time when analyzing multiple datasets.

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
lifetimes. Multi-component analysis can be done in the "Components" tab of the
"Phasor Plot" widget, either by projection to a line between components (only two
component analysis), or by component fitting 'n' number of components based on the
available harmonics. For more than three component analysis, the location of the
components in higher harmonics must be provided.

![components](https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/components.gif)

### Apparent or Normal Lifetime Analysis

A FLIM image can be colormapped according to the phase or modulation apparent 
lifetime, as well as the normal lifetime in the "Lifetime" tab of the "Phasor 
Plot" widget. A histogram is also created for visualization of the distribution
of apparent lifetimes of the FLIM image.

![lifetimes](https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/lifetime.gif)

### FRET Analysis

Analyze Förster Resonance Energy Transfer (FRET) trajectories and efficiencies
in the "FRET" tab of the "Phasor Plot" widget. The lifetime of the donor and the location of the
background in the phasor plot can be obtained automatically from another layer.

![fret](https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/fret.gif)


### Phasor Custom Import

Supported file formats (`.tif`, `.ptu`, `.sdt`, `.fbd`, `.lsm`, `.ome.tif`) 
can be imported and converted to phasor space using the "Phasor Custom Import" widget.
Depending on the file format, you can specify additional options such as harmonics, channels, and frames.
The signal can then be visualized according to the selected parameters for each file.

![custom_import](https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/import.gif)

### Data Export

The average intensity image and phasor coordinates can be exported as OME-TIF files, which are compatible with both napari-phasors and PhasorPy. Alternatively, you can export the phasor coordinates and selections as a CSV file using the "Export Phasor" widget. Analysis results—such as lifetime, FRET efficiency, and component fractions—can also be exported to CSV. Additionally, the colormapped image layer can be exported with or without its associated colorbar.

![export_phasors](https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/export.gif)

## Installation

You can install `napari-phasors` via [pip]. Follow these steps from a 
terminal.

We recommend using `miniforge` whenever possible. Click 
[here](https://conda-forge.org/download/) to choose the right download option for your OS.
**If you do not use `miniforge`, but rather Anaconda or Miniconda, replace 
the `mamba` term whenever you see it below with `conda`.**

Create a conda environment with napari by typing :

    mamba create -y -n napari-phasors-env napari pyqt python=3.12
    
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
"napari-phasors" is free and open source software.

Please cite doi: [https://doi.org/10.5281/zenodo.14647626](https://doi.org/10.5281/zenodo.14647626) if napari-phasors contributes to a project that leads to a publication.

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
