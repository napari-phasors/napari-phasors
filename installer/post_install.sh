#!/bin/bash
# Post-install script: create launchers. napari-phasors and its dependencies
# are bundled into the env by constructor (see specs in the build workflow),
# so nothing is downloaded here and the install needs no network.

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
# Shield the bundled python from user-level python config: a stray
# PYTHONPATH or ~/.local/lib/pythonX.Y/site-packages with incompatible
# packages (numpy, Qt bindings, ...) would be imported into the app and
# can crash it on startup.
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
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
if [ "$(uname)" = "Linux" ]; then
    # Recent Qt may auto-select the Wayland backend on Wayland sessions;
    # napari is only reliable on xcb, and XWayland covers Wayland desktops.
    # Respect an explicit user override.
    export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
    # The .desktop entry launches with Terminal=false, so any startup
    # error would otherwise be invisible. When not attached to a
    # terminal, keep a log of the last launch attempt.
    if [ ! -t 1 ]; then
        LOG_FILE="${DIR}/last-launch.log"
        if ! : > "${LOG_FILE}" 2>/dev/null; then
            LOG_FILE="${TMPDIR:-/tmp}/napari-phasors-launch.log"
            : > "${LOG_FILE}" 2>/dev/null || LOG_FILE=/dev/null
        fi
        exec >> "${LOG_FILE}" 2>&1
        echo "napari-phasors launch: $(date)"
    fi
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
