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
and hyperspectral file formats, performing phasor analysis on multiple layers
simultaneously, calibration, component analysis, FRET analysis, filtering,
phasor selections (manual, circular cursor, and automatic clustering), and
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

### Multiple-Layer Selection and Simultaneous Analysis

Multiple image layers containing phasor data can be selected simultaneously
from the layer dropdown in the "Phasor Plot" widget. All layers can be
selected or deselected at once using the "All" and "None" controls. When more
than one layer is selected, their phasor coordinates are merged and displayed
together in the phasor plot, enabling direct comparison and joint analysis. A
primary layer can be designated from the same dropdown to drive plot settings
and analysis parameters (such as calibration, frequency, filter settings, and
component locations) for all selected layers. All analyses are then applied to
every selected layer at once.

![multiple_layers](https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/multiple%20layers.gif)

### Phasor Calibration

FLIM images can be calibrated using a reference image acquired under the same
experimental parameters in the "Calibration" tab of the "Phasor Plot" widget.
This reference image should consist of a homogeneous solution of a fluorophore
with a known fluorescence lifetime and the laser frequency used in the
experiment. This ensures accuracy and consistency in lifetime measurements.

![calibration](https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/calibration.gif)

### Filtering and Thresholding

Apply various filters and thresholds to your phasor data to enhance analysis
quality in the "Filter" tab on the "Phasor Plot" widget. You can
filter phasor coordinates using the median or wavelet filter.

![filter_threshold](https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/filter%20threshold.gif)

### Mask

You can create a mask using either a shapes layer or a labels layer in napari. Once the
mask is created, select it from the mask combobox in the "Phasor Plot" widget. Only the
pixels inside the selected mask will be plotted in the phasor space and included in subsequent
analyses. This allows you to focus your analysis on specific regions of interest within your data.

![mask](https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/mask.gif)

### Copy Settings and Analysis

You can quickly copy plot settings and analysis parameters, such as calibration, frequency, filter
settings, and component locations, from another layer or from an OME-TIF file previously exported
with the napari-phasors plugin. This feature streamlines the workflow by allowing you to reuse
established configurations, ensuring consistency and saving time when analyzing multiple datasets.

![copy_settings](https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/copy%20settings.gif)

### Phasor Selections

The "Selection" tab of the "Phasor Plot" widget offers three modes for
identifying regions of interest in the phasor plot and mapping the
corresponding pixels back to the intensity image.

#### Circular Cursor Selection

Define one or more circular cursors on the phasor plot to select regions of
interest. Each cursor can be positioned and resized interactively by dragging
it on the plot, or by entering coordinates directly in the table. A separate
labels layer is created for each cursor, color-coded for easy identification.
Statistics for each cursor region are displayed in the table.

![circular_cursors](https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/circular%20cursors.gif)

#### Automatic Clustering

Automatically segment the phasor plot into clusters using Gaussian Mixture
Models (GMM). The number of clusters can be specified, and the resulting
clusters are displayed as ellipses on the phasor plot. Each cluster is
assigned a color that can be customized, and the corresponding pixels are
highlighted in a labels layer. Cluster statistics are shown in the table.

![automatic_clustering](https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/automatic%20clustering.gif)

#### Manual Selection

Draw free-form selections directly on the phasor plot using different shape
tools available at the top of the plot. The selection ID can be managed from
the "Selection" tab, allowing multiple independent selections to be stored
and toggled. NOTE: Manual selections are not stored when exporting in OME-TIF format.

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

The average intensity image and phasor coordinates can be exported as OME-TIF files, which are compatible
with both napari-phasors and PhasorPy. Alternatively, you can export the phasor coordinates and selections
as a CSV file using the "Export Phasor" widget. Analysis results—such as lifetime, FRET efficiency, and
component fractions—can also be exported to CSV. Additionally, the colormapped image layer can be exported
with or without its associated colorbar.

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

### Pre-commit Hooks

This project uses [pre-commit](https://pre-commit.com/) to run **black**,
**isort**, and **ruff** automatically on every commit. To set it up:

```bash
pip install pre-commit
pre-commit install
```

From now on, every `git commit` will auto-format and lint your code before
the commit goes through. You can also run the hooks manually on all files:

```bash
pre-commit run --all-files
```

## License

Distributed under the terms of the [BSD-3] license,
"napari-phasors" is free and open source software.

Please cite doi: [https://doi.org/10.5281/zenodo.14647626](https://doi.org/10.5281/zenodo.14647626)
if napari-phasors contributes to a project that leads to a publication.

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
