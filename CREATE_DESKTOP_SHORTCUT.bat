@echo off
chcp 65001 >nul
title AIFACE - Create Desktop Shortcut
cd /d "%~dp0"

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ws = New-Object -ComObject WScript.Shell;" ^
  "$desktop = [Environment]::GetFolderPath('Desktop');" ^
  "$lnk = $ws.CreateShortcut((Join-Path $desktop 'AIFACE.lnk'));" ^
  "$lnk.TargetPath = (Join-Path '%~dp0' 'START_AIFACE.bat');" ^
  "$lnk.WorkingDirectory = '%~dp0'.TrimEnd('\');" ^
  "$lnk.IconLocation = 'shell32.dll,277';" ^
  "$lnk.Description = 'AIFACE - AI Face Recognition System';" ^
  "$lnk.Save();" ^
  "Write-Host ''; Write-Host '  Shortcut created: ' (Join-Path $desktop 'AIFACE.lnk')"

echo.
echo  Double-click the AIFACE icon on your Desktop to start the system.
echo.
pause
