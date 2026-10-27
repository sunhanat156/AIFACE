@echo off
chcp 65001 >nul
title AIFACE - AI Face Recognition
cd /d "%~dp0"

set "PYTHONIOENCODING=utf-8"
set "PYTHON_CMD="

rem Prefer a project virtual environment if one exists
if exist "%~dp0.venv\Scripts\python.exe" set "PYTHON_CMD=%~dp0.venv\Scripts\python.exe"
if exist "%~dp0venv\Scripts\python.exe"  set "PYTHON_CMD=%~dp0venv\Scripts\python.exe"

if not defined PYTHON_CMD (
    where python >nul 2>nul && set "PYTHON_CMD=python"
)
if not defined PYTHON_CMD (
    where py >nul 2>nul && set "PYTHON_CMD=py"
)

if not defined PYTHON_CMD (
    echo.
    echo  [ERROR] Python was not found on this computer.
    echo.
    echo  Install Python 3.12 from https://www.python.org/downloads/
    echo  and be sure to tick "Add python.exe to PATH" during setup.
    echo.
    pause
    exit /b 1
)

"%PYTHON_CMD%" "%~dp0launcher.py"

echo.
echo  AIFACE has stopped. Press any key to close this window.
pause >nul
