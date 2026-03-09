REM Post-install script: install napari-phasors and create shortcuts
call "%PREFIX%\Scripts\pip.exe" install napari-phasors

REM Create a launcher batch file
echo @echo off > "%PREFIX%\napari-phasors.bat"
echo set "PATH=%PREFIX%;%PREFIX%\Library\bin;%PREFIX%\Scripts;%%PATH%%" >> "%PREFIX%\napari-phasors.bat"
echo start "" "%PREFIX%\Scripts\napari.exe" >> "%PREFIX%\napari-phasors.bat"

REM Create shortcuts via VBScript (always available, unlike PowerShell in NSIS)
echo Set ws = CreateObject("WScript.Shell") > "%PREFIX%\create_shortcuts.vbs"
echo Set desktop = ws.CreateShortcut(ws.SpecialFolders("Desktop") ^& "\napari-phasors.lnk") >> "%PREFIX%\create_shortcuts.vbs"
echo desktop.TargetPath = "%PREFIX%\napari-phasors.bat" >> "%PREFIX%\create_shortcuts.vbs"
echo desktop.WorkingDirectory = "%USERPROFILE%" >> "%PREFIX%\create_shortcuts.vbs"
echo desktop.Description = "napari-phasors" >> "%PREFIX%\create_shortcuts.vbs"
echo desktop.WindowStyle = 7 >> "%PREFIX%\create_shortcuts.vbs"
echo desktop.Save >> "%PREFIX%\create_shortcuts.vbs"
echo Set startmenu = ws.CreateShortcut(ws.SpecialFolders("StartMenu") ^& "\napari-phasors.lnk") >> "%PREFIX%\create_shortcuts.vbs"
echo startmenu.TargetPath = "%PREFIX%\napari-phasors.bat" >> "%PREFIX%\create_shortcuts.vbs"
echo startmenu.WorkingDirectory = "%USERPROFILE%" >> "%PREFIX%\create_shortcuts.vbs"
echo startmenu.Description = "napari-phasors" >> "%PREFIX%\create_shortcuts.vbs"
echo startmenu.WindowStyle = 7 >> "%PREFIX%\create_shortcuts.vbs"
echo startmenu.Save >> "%PREFIX%\create_shortcuts.vbs"
cscript //nologo "%PREFIX%\create_shortcuts.vbs"
del "%PREFIX%\create_shortcuts.vbs"
