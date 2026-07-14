#!/bin/bash
# Post-install script: install napari-phasors and create launchers
"${PREFIX}/bin/pip" install napari-phasors

# Create a launcher script. Activate the conda environment before starting
# napari so Qt can find its platform plugins and libraries. Without
# activation, the environment's activate.d scripts (which set QT_PLUGIN_PATH,
# FONTCONFIG_PATH, GSETTINGS_SCHEMA_DIR, LD_LIBRARY_PATH, ...) never run and
# napari aborts on launch with "could not load the Qt platform plugin 'xcb'".
cat > "${PREFIX}/napari-phasors" << 'LAUNCHER'
#!/bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${DIR}/bin/activate" "${DIR}"
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
