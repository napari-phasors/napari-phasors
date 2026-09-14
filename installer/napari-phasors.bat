@echo off
REM Launcher for the bundled napari-phasors application (Windows).
REM Installed at the root of the constructor prefix and pointed at by the
REM desktop and Start Menu shortcuts created in post_install.bat.
setlocal
call "%~dp0launcher-env.bat"
start "" "%NAPARI_PHASORS_ENV%\Scripts\napari.exe" %*
endlocal
