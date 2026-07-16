#!/bin/bash
# Post-install script: install napari-phasors where it cannot be bundled,
# then create launchers.
#
# Linux: napari-phasors and its dependencies are bundled into the env by
# constructor (see specs in the build workflow), so nothing is downloaded
# here and the install needs no network.
#
# macOS: phasorpy, fbdfile and ptufile have no osx-arm64 packages on
# conda-forge, so napari-phasors cannot be part of the solved env and is
# installed from PyPI here instead. This runs on the CI runner rather than on
# a user machine: the DMG step executes this installer and packages the
# resulting env into the .app, so the DMG users download is still
# self-contained.
if [ "$(uname)" = "Darwin" ]; then
    "${PREFIX}/bin/pip" install "napari-phasors${INSTALLER_VER:+==${INSTALLER_VER}}"
fi

# Create a launcher script. Constructor envs do not ship bin/activate
# (no conda inside), so export what napari/Qt need directly: the env's
# bin dir on PATH and explicit Qt plugin/fontconfig paths. Without these,
# Qt cannot find its platform plugin and napari aborts silently (the
# .desktop file uses Terminal=false, so no error is ever shown).
cat > "${PREFIX}/napari-phasors" << 'LAUNCHER'
#!/bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="${DIR}/bin:${PATH}"
export CONDA_PREFIX="${DIR}"
for PLUGIN_DIR in "${DIR}/lib/qt6/plugins" "${DIR}/lib/qt/plugins" "${DIR}/plugins"; do
    if [ -d "${PLUGIN_DIR}/platforms" ]; then
        export QT_PLUGIN_PATH="${PLUGIN_DIR}"
        export QT_QPA_PLATFORM_PLUGIN_PATH="${PLUGIN_DIR}/platforms"
        break
    fi
done
if [ -d "${DIR}/etc/fonts" ]; then
    export FONTCONFIG_PATH="${DIR}/etc/fonts"
fi
exec "${DIR}/bin/python" -m napari "$@"
LAUNCHER
chmod +x "${PREFIX}/napari-phasors"

# Create a .desktop file for Linux (freedesktop standard)
if [ "$(uname)" = "Linux" ]; then
    DESKTOP_DIR="${HOME}/Desktop"
    APPS_DIR="${HOME}/.local/share/applications"
    mkdir -p "${APPS_DIR}"

    cat > "${APPS_DIR}/napari-phasors.desktop" << DESKTOP
[Desktop Entry]
Type=Application
Name=napari-phasors
Comment=Phasor analysis in napari
Exec=${PREFIX}/napari-phasors
Icon=${PREFIX}/icon.png
Terminal=false
Categories=Science;Education;
DESKTOP

    # Also copy to Desktop if it exists
    if [ -d "${DESKTOP_DIR}" ]; then
        cp "${APPS_DIR}/napari-phasors.desktop" "${DESKTOP_DIR}/"
        chmod +x "${DESKTOP_DIR}/napari-phasors.desktop"
    fi
fi

# macOS: no additional launchers needed (handled by .app bundle in DMG)
if [ "$(uname)" = "Darwin" ]; then
    echo "napari-phasors installed successfully."
fi
