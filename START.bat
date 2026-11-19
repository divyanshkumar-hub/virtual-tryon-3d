@echo off
title Virtual Try-On 3D Studio — Launcher
color 0A

echo ============================================================
echo   AI Fashion Stylist ^& 3D Virtual Try-On Studio
echo   Phase 1 — Full Integrated System
echo ============================================================
echo.

cd /d "%~dp0"

echo [1/3] Checking Python...
py --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.x from python.org
    pause
    exit /b 1
)
py --version

echo.
echo [2/3] Installing / verifying Python dependencies...
py -m pip install flask flask-cors flask-sqlalchemy werkzeug pillow numpy --quiet --disable-pip-version-check
echo      Dependencies OK!

echo.
echo [3/3] Starting StyleMate Flask Server on port 5000...
echo       The app will open in your browser automatically.
echo       Press Ctrl+C to stop the server.
echo.
echo ============================================================

py mini.py

pause
