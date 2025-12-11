(intro)=

# napari-phasors

napari-phasors is a comprehensive plugin, based on the [phasorpy](https://www.phasorpy.org/) library, that provides a complete workflow for phasor analysis in napari. It includes widgets for reading various FLIM and hyperspectral file formats, performing phasor analysis, calibration, component analysis, FRET analysis, filtering, manual selections, and exporting results.

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

[github]: https://github.com/napari-phasors/napari-phasors "GitHub source code repository for this project"
[tutorial]: https://docs.readthedocs.io/en/stable/tutorial/index.html "Official Read the Docs Tutorial"
[jb-docs]: https://jupyterbook.org/en/stable/ "Official Jupyter Book documentation"