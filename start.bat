@echo off
title Stupid Bot Manager
echo [1/2] Starting NSFW Scanner API...
start "NSFW Scanner" cmd /k "python scanner_api.py"

echo [2/2] Starting Discord Bot...
python main.py
pause
