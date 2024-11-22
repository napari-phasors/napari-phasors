# napari-phasors

[![License BSD-3](https://img.shields.io/pypi/l/napari-phasors.svg?color=green)](https://github.com/napari-phasors/napari-phasors/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-phasors.svg?color=green)](https://pypi.org/project/napari-phasors)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-phasors.svg?color=green)](https://python.org)
[![tests](https://github.com/napari-phasors/napari-phasors/workflows/tests/badge.svg)](https://github.com/napari-phasors/napari-phasors/actions)
[![codecov](https://codecov.io/gh/napari-phasors/napari-phasors/branch/main/graph/badge.svg)](https://codecov.io/gh/napari-phasors/napari-phasors)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-phasors)](https://napari-hub.org/plugins/napari-phasors)

A simple plugin to do phasor analysis in napari. Based on the [phasorpy](https://www.phasorpy.org/) library.

[Jump to Intallation](#installation)

----------------------------------

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->

## Usage

napari-phasors is composed of a few widgets that allow reading a few specific FLIM and hyperspectral file formats, perform phasor analysis, and display and export the results of manual phasor selections.

### Sample Data

![sample_data](https://github.com/napari-phasors/napari-phasors/raw/main/gifs/sample_data.gif)

### Phasor Analysis

#### FLIM Data

![phasors_flim](https://github.com/napari-phasors/napari-phasors/raw/main/gifs/phasors_flim.gif)

#### Hyperspectral Data

![phasors_hyperspectral](https://github.com/napari-phasors/napari-phasors/raw/main/gifs/phasors_hyperspectral.gif)

### Apparent Lifetime Display

![lifetimes](https://github.com/napari-phasors/napari-phasors/raw/main/gifs/lifetimes.gif)

### Phasor Calibration

![calibration](https://github.com/napari-phasors/napari-phasors/raw/main/gifs/calibration.gif)

### Phasor Custom Import

![custom_import](https://github.com/napari-phasors/napari-phasors/raw/main/gifs/custom_import.gif)

### Phasor Export

![export_phasors](https://github.com/napari-phasors/napari-phasors/raw/main/gifs/export_phasors.gif)

## Installation

You can install `napari-phasors` via [pip]. Follow these steps from a terminal.

We recommend using `miniforge` whenever possible. Click [here](https://github.com/conda-forge/miniforge?tab=readme-ov-file#miniforge) to choose the right download option for your OS.
**If you do not use `miniforge`, but rather Anaconda or Miniconda, replace the `mamba` term whenever you see it below with `conda`.**

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

If you encounter any problems, please [file an issue] along with a detailed description.

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
