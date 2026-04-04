# napari-phasors

[![License BSD-3](https://img.shields.io/pypi/l/napari-phasors.svg?color=green)](https://github.com/napari-phasors/napari-phasors/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-phasors.svg?color=green)](https://pypi.org/project/napari-phasors)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-phasors.svg?color=green)](https://python.org)
[![tests](https://github.com/napari-phasors/napari-phasors/actions/workflows/run-tests.yml/badge.svg?branch=main)](https://github.com/napari-phasors/napari-phasors/actions/workflows/run-tests.yml)
[![codecov](https://codecov.io/gh/napari-phasors/napari-phasors/branch/main/graph/badge.svg)](https://codecov.io/gh/napari-phasors/napari-phasors)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-phasors)](https://napari-hub.org/plugins/napari-phasors)
[![Documentation Status](https://readthedocs.org/projects/napari-phasors/badge/?version=latest)](https://napari-phasors.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.14647626-blue.svg)](https://doi.org/10.5281/zenodo.14647626)

A comprehensive plugin for phasor analysis in napari. Based on the
[phasorpy](https://www.phasorpy.org/) library.

----------------------------------

## Documentation

Full documentation, including step-by-step guides and the API reference, is
available at **[https://napari-phasors.readthedocs.io](https://napari-phasors.readthedocs.io)**.

## Features

- Read FLIM and hyperspectral files (`.fbd`, `.sdt`, `.ptu`, `.lsm`, `.ome.tif`)
- Simultaneous multi-layer phasor analysis with primary-layer selection
- Calibration, filtering (median / wavelet), and automatic thresholding
- Phasor selections: circular cursors, automatic clustering, and manual drawing
- Phasor Mapping: colormap apparent/normal lifetime, phasor phase, and modulation per pixel, with an interactive 1D histogram and statistics table
- Component analysis and FRET trajectory analysis
- Export as OME-TIF or CSV (multiple layers simultaneously)

## Installation

You can install `napari-phasors` via [pip]. Follow these steps from a
terminal.

We recommend using `miniforge` whenever possible. Click
[here](https://conda-forge.org/download/) to choose the right download option for your OS.
**If you do not use `miniforge`, but rather Anaconda or Miniconda, replace
the `mamba` term whenever you see it below with `conda`.**

Create a conda environment with napari by typing :

    mamba create -y -n napari-phasors-env napari pyqt6 python=3.14 # or 3.12 or 3.13

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
