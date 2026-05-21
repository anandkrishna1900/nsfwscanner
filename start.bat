@echo off
title NSFW Scanner Bot
if not exist "venv\Scripts\python.exe" (
    echo Virtual environment not found.
    echo Run install_all_dependencies.bat first.
    pause
    exit /b 1
)
echo Starting Discord Bot...
"venv\Scripts\python.exe" main.py
pause
