@echo off
title StyleMate v2 Integrated Full-Stack Server (Port 5000)
cd /d "%~dp0"
echo ========================================================
echo   Starting StyleMate v2 Full-Stack Server
echo   Integrated React AI Stylist, 3D Try-On & REST API
echo.
echo   Main Application:      http://127.0.0.1:5000
echo   Standalone 3D Studio:  http://127.0.0.1:5000/studio
echo ========================================================
py app.py
pause
