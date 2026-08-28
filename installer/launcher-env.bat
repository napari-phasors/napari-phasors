@echo off
REM Shared environment setup for the bundled napari-phasors application.
REM
REM `call` this from a batch file; it is the single source of truth for
REM everything the launcher needs, and CI reuses it so the smoke tests
REM exercise exactly the environment a user gets.
REM
REM The bundle uses a two-environment layout (see make_construct.py):
REM
REM     <root>\                     base env: python + conda tooling
REM     <root>\envs\napari-phasors  app env:  napari + napari-phasors + Qt

set "NAPARI_PHASORS_ROOT=%~dp0"
if "%NAPARI_PHASORS_ROOT:~-1%"=="\" set "NAPARI_PHASORS_ROOT=%NAPARI_PHASORS_ROOT:~0,-1%"
set "NAPARI_PHASORS_ENV=%NAPARI_PHASORS_ROOT%\envs\napari-phasors"

REM Same order `conda activate` uses for the app env, then the base env dirs
REM so the bundled conda tooling is reachable. Without Library\bin on PATH,
REM the Qt DLLs the PyQt6 extension modules link against cannot be resolved.
set "PATH=%NAPARI_PHASORS_ENV%;%NAPARI_PHASORS_ENV%\Library\mingw-w64\bin;%NAPARI_PHASORS_ENV%\Library\usr\bin;%NAPARI_PHASORS_ENV%\Library\bin;%NAPARI_PHASORS_ENV%\Scripts;%NAPARI_PHASORS_ENV%\bin;%NAPARI_PHASORS_ROOT%;%NAPARI_PHASORS_ROOT%\Library\bin;%NAPARI_PHASORS_ROOT%\Scripts;%NAPARI_PHASORS_ROOT%\condabin;%PATH%"
set "CONDA_PREFIX=%NAPARI_PHASORS_ENV%"

REM Shield the bundled python from user-level python config.
set "PYTHONPATH="
set "PYTHONHOME="
set "PYTHONNOUSERSITE=1"

REM napari-plugin-manager resolves its conda backend from MAMBA_EXE, CONDA_EXE
REM and CONDA in that order, falling back to whatever `conda.bat` is on PATH.
REM Point it at the bundled conda and clear the others, so a conda installation
REM that happens to be active in the user's shell can never operate on this
REM prefix. A real .exe is used rather than condabin\conda.bat because QProcess
REM starts programs through CreateProcess, which cannot execute batch files.
REM CONDARC is cleared for the same reason: the bundled <root>\.condarc must
REM be what plugin installs are solved against.
set "MAMBA_EXE="
set "CONDA="
set "CONDA_EXE="
set "CONDARC="
REM Preferred: the conda in the base environment.
if exist "%NAPARI_PHASORS_ROOT%\Scripts\conda.exe" set "CONDA_EXE=%NAPARI_PHASORS_ROOT%\Scripts\conda.exe"
REM Fallback: the conda-standalone binary constructor leaves behind for its
REM own uninstaller.
if not defined CONDA_EXE if exist "%NAPARI_PHASORS_ROOT%\_conda.exe" set "CONDA_EXE=%NAPARI_PHASORS_ROOT%\_conda.exe"

REM The bundle ships PyQt6. Pin qtpy to it so that a plugin pulling in another
REM binding later cannot change which one napari imports.
set "QT_API=pyqt6"
