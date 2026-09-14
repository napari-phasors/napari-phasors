#!/bin/bash
# Post-install script: make the shipped launchers executable and register the
# Linux desktop entry. napari-phasors and all of its dependencies are bundled
# into the environments by constructor (see make_construct.py), so nothing is
# downloaded here and the install needs no network.
#
# The launcher itself (napari-phasors), its environment setup
# (launcher-env.sh) and the relocatable conda entry point (conda-launcher)
# ship as constructor `extra_files`, which do not carry permission bits.

for launcher in "${PREFIX}/napari-phasors" "${PREFIX}/conda-launcher"; do
    if [ ! -f "${launcher}" ]; then
        echo "ERROR: missing bundled launcher: ${launcher}" >&2
        exit 1
    fi
    chmod +x "${launcher}"
done

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
Exec="${PREFIX}/napari-phasors"
Icon=${PREFIX}/icon.png
Terminal=false
StartupNotify=true
StartupWMClass=napari
Categories=Science;Education;
DESKTOP
    update-desktop-database "${APPS_DIR}" >/dev/null 2>&1 || true

    # Also copy to Desktop if it exists. GNOME refuses to launch desktop
    # icons that are not both executable and marked trusted; without the
    # gio metadata a double-click does nothing (or shows an "untrusted
    # launcher" warning), which looks like the app never opens.
    if [ -d "${DESKTOP_DIR}" ]; then
        cp "${APPS_DIR}/napari-phasors.desktop" "${DESKTOP_DIR}/"
        chmod +x "${DESKTOP_DIR}/napari-phasors.desktop"
        gio set "${DESKTOP_DIR}/napari-phasors.desktop" \
            metadata::trusted true >/dev/null 2>&1 || true
    fi

    echo "napari-phasors installed successfully."
    echo "Launch it from your applications menu, or run:"
    echo "    ${PREFIX}/napari-phasors"
    echo "If the app does not open, check the log at:"
    echo "    ${PREFIX}/last-launch.log"
fi

# macOS: no additional launchers needed (handled by .app bundle in DMG)
if [ "$(uname)" = "Darwin" ]; then
    echo "napari-phasors installed successfully."
fi
