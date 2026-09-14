# Shared environment setup for the bundled napari-phasors application.
#
# Source this from bash; it is the single source of truth for everything the
# launcher needs, and CI reuses it so the smoke tests exercise exactly the
# environment a user gets.
#
# The bundle uses a two-environment layout (see make_construct.py):
#
#     <root>/                     base env: python + conda tooling
#     <root>/envs/napari-phasors  app env:  napari + napari-phasors + Qt
#
# NAPARI_PHASORS_ROOT may be preset by the caller; otherwise it is derived
# from this file's own location, which keeps the bundle relocatable (the
# macOS .app moves the whole tree after installation).

NAPARI_PHASORS_ROOT="${NAPARI_PHASORS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
NAPARI_PHASORS_ENV="${NAPARI_PHASORS_ROOT}/envs/napari-phasors"
export NAPARI_PHASORS_ROOT NAPARI_PHASORS_ENV

# Shield the bundled python from user-level python config: a stray PYTHONPATH
# or ~/.local/lib/pythonX.Y/site-packages with incompatible packages (numpy,
# Qt bindings, ...) would be imported into the app and can crash it on startup.
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1

# App env first (that is what napari runs from), then the base env so the
# bundled conda tooling is reachable.
export PATH="${NAPARI_PHASORS_ENV}/bin:${NAPARI_PHASORS_ROOT}/bin:${PATH}"
export CONDA_PREFIX="${NAPARI_PHASORS_ENV}"

# napari-plugin-manager resolves its conda backend from MAMBA_EXE, CONDA_EXE
# and CONDA in that order, falling back to whatever `conda` is on PATH. Point
# it at the bundled conda and clear the others, so a conda installation that
# happens to be active in the user's shell can never operate on this prefix.
# CONDARC is cleared for the same reason: the bundled <root>/.condarc must be
# what plugin installs are solved against.
unset MAMBA_EXE CONDA CONDA_EXE CONDARC
if [ -x "${NAPARI_PHASORS_ROOT}/conda-launcher" ]; then
    export CONDA_EXE="${NAPARI_PHASORS_ROOT}/conda-launcher"
elif [ -x "${NAPARI_PHASORS_ROOT}/_conda" ]; then
    # Fallback: the conda-standalone binary constructor leaves behind for its
    # own uninstaller.
    export CONDA_EXE="${NAPARI_PHASORS_ROOT}/_conda"
fi

# Conda bakes the build-time install prefix into the env, so Qt's compiled-in
# plugin path points at a directory that may no longer exist. Point Qt at the
# bundled plugins explicitly - env vars override the baked-in path. Without
# these, Qt cannot find its platform plugin and napari aborts silently.
for _np_plugin_dir in "${NAPARI_PHASORS_ENV}/lib/qt6/plugins" \
                      "${NAPARI_PHASORS_ENV}/lib/qt/plugins" \
                      "${NAPARI_PHASORS_ENV}/plugins"; do
    if [ -d "${_np_plugin_dir}/platforms" ]; then
        export QT_PLUGIN_PATH="${_np_plugin_dir}"
        export QT_QPA_PLATFORM_PLUGIN_PATH="${_np_plugin_dir}/platforms"
        break
    fi
done
unset _np_plugin_dir
if [ -d "${NAPARI_PHASORS_ENV}/etc/fonts" ]; then
    export FONTCONFIG_PATH="${NAPARI_PHASORS_ENV}/etc/fonts"
fi

# The bundle ships PyQt6. Pin qtpy to it so that a plugin pulling in another
# binding later cannot change which one napari imports.
export QT_API=pyqt6

if [ "$(uname)" = "Linux" ]; then
    # Recent Qt may auto-select the Wayland backend on Wayland sessions;
    # napari is only reliable on xcb, and XWayland covers Wayland desktops.
    # Respect an explicit user override.
    export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
fi
