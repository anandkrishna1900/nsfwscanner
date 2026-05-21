@echo off
setlocal
title NSFW Scanner Bot - Dependency Installer

echo.
echo Installing NSFW Scanner Bot dependencies...
echo.

where py >nul 2>nul
if not %errorlevel%==0 (
    echo ERROR: Python Launcher was not found.
    echo Install Python 3.11 from https://www.python.org/downloads/ and enable "Add Python to PATH".
    echo.
    pause
    exit /b 1
)

py -3.11 --version >nul 2>nul
if not %errorlevel%==0 (
    echo ERROR: Python 3.11 was not found.
    echo This bot should use Python 3.11 because ML packages may not support Python 3.14 yet.
    echo Install Python 3.11, then run this installer again.
    echo.
    pause
    exit /b 1
)

set "PYTHON_CMD=py -3.11"

if not exist "requirements.txt" (
    echo ERROR: requirements.txt was not found in this folder.
    echo Run this installer from the project root.
    echo.
    pause
    exit /b 1
)

if not exist "venv\Scripts\python.exe" (
    echo Creating virtual environment...
    %PYTHON_CMD% -m venv venv
    if errorlevel 1 (
        echo.
        echo ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
) else (
    echo Using existing virtual environment.
)

echo.
echo Upgrading pip...
"venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 (
    echo.
    echo ERROR: Failed to upgrade pip.
    pause
    exit /b 1
)

echo.
echo Installing packages from requirements.txt...
"venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo ERROR: Dependency installation failed.
    echo Check the error above, then run this file again.
    pause
    exit /b 1
)

echo.
echo Dependencies installed successfully.
echo You can now configure .env and run start.bat.
echo.
pause
