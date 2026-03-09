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

- **Reading** a wide range of FLIM and hyperspectral file formats:
    - **Raw data formats:**
        - `.ptu`, `.fbd`, `.sdt`, `.lsm`, `.tif`, `.tiff`, `.czi`, `.flif`, `.bh`, `.b&h`, `.bhz`, `.lif`, `.bin`, `.json`
    - **Processed data formats:**
        - `.ome.tif`, `.ome.tiff`, `.r64`, `.ref`, `.ifli`, `.lif`, `.json`
- **Phasor analysis** on multiple layers simultaneously, including support for stacking multiple raw data files
- **Calibration** using reference images with known lifetimes
- **Filtering** with median, wavelet, and automatic thresholding (Otsu, Li, Yen)
- **Component analysis** for multi-component systems
- **Phasor Mapping** — colormap apparent/normal lifetime, phasor phase, and phasor modulation per pixel, with interactive 1D histograms, statistics tables, and arc overlay tools
- **FRET analysis** with donor trajectory visualization and multi-layer donor/background source selection
- **Selections** via manual drawing, circular/polar/elliptical cursors, and automatic clustering
- **Exporting** results as OME-TIF or CSV (multiple layers simultaneously)

## Installation

There are several ways to install napari-phasors:

### Easiest: napari's plugin manager

If you already have [napari](https://napari.org) installed, go to
**Plugins → Install/Uninstall Plugins...**, search for **napari-phasors**,
and click **Install**.

### Standalone installer (no Python required)

Pre-built installers for Windows, macOS, and Linux are available on the
[Releases](https://github.com/napari-phasors/napari-phasors/releases) page.
Download and run — no Python installation needed.

### Using conda + pip

We recommend using [miniforge](https://conda-forge.org/download/). If you
use Anaconda or Miniconda, replace `mamba` with `conda`.

    mamba create -y -n napari-phasors-env napari pyqt6 python=3.14 # or 3.12 or 3.13

Activate the environment :

    mamba activate napari-phasors-env
    pip install napari-phasors

### Development version

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
