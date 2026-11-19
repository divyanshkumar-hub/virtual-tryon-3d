@echo off
title StyleMate v2 - Virtual Try-On Studio
color 0B

echo ==============================================================
echo    StyleMate v2 — AI Fashion Stylist ^& 3D Virtual Try-On
echo ==============================================================
echo.

cd /d "%~dp0"

echo [1/3] Verifying Python environment...
py --version >nul 2>&1
if errorlevel 1 (
    echo Python not found. Please install Python 3.10+
    pause
    exit /b 1
)

echo [2/3] Checking dependencies...
py -m pip install -r requirements.txt --quiet --disable-pip-version-check
echo       Dependencies ready.

echo [3/3] Starting StyleMate v2 on http://127.0.0.1:5000...
echo       Press Ctrl+C to terminate.
echo ==============================================================
echo.

start "" "http://127.0.0.1:5000"
py app.py

pause
