# Installation

## Prerequisites

- Python 3.12 or 3.13
- A working Qt backend (PyQt5 is installed automatically)

## Using conda + pip (recommended)

We recommend using [miniforge](https://conda-forge.org/download/).
If you use Anaconda or Miniconda, replace `mamba` with `conda`.

Create an environment with napari:

```bash
mamba create -y -n napari-phasors-env napari pyqt python=3.12 # or 3.13
mamba activate napari-phasors-env
```

Install napari-phasors:

```bash
pip install napari-phasors
```

## Development installation

Clone the repository and install in editable mode with testing dependencies:

```bash
git clone https://github.com/napari-phasors/napari-phasors.git
cd napari-phasors
pip install -e ".[testing]"
```

### Pre-commit hooks

The project uses [pre-commit](https://pre-commit.com/) to enforce **black**, **isort**, and **ruff** on every commit:

```bash
pip install pre-commit
pre-commit install
```

### Running tests

```bash
pytest src/napari_phasors/_tests/
```

Or in parallel:

```bash
pytest src/napari_phasors/_tests/ -n auto
```
