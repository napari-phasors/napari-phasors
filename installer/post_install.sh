#!/bin/bash
# Post-install script: install napari-phasors and create launchers
"${PREFIX}/bin/pip" install napari-phasors

# Create a launcher script
cat > "${PREFIX}/napari-phasors" << 'LAUNCHER'
#!/bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="${DIR}/bin:${DIR}/lib:${PATH}"
exec "${DIR}/bin/napari" "$@"
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
Terminal=false
Categories=Science;Education;
DESKTOP

    # Also copy to Desktop if it exists
    if [ -d "${DESKTOP_DIR}" ]; then
        cp "${APPS_DIR}/napari-phasors.desktop" "${DESKTOP_DIR}/"
        chmod +x "${DESKTOP_DIR}/napari-phasors.desktop"
    fi
fi

# Create an alias-style launcher for macOS
if [ "$(uname)" = "Darwin" ]; then
    # Create a simple command-line launcher in /usr/local/bin if writable
    if [ -w "/usr/local/bin" ]; then
        ln -sf "${PREFIX}/napari-phasors" "/usr/local/bin/napari-phasors"
    fi
    echo ""
    echo "=== napari-phasors installed ==="
    echo "Run with: ${PREFIX}/napari-phasors"
    echo ""
fi
