@echo off
title StyleMate Flask + SQLite + AI 3D Backend (Port 5000)
color 0b
echo ====================================================================
echo     STYLEMATE FLASK + SQLITE DB + AI 3D BACKEND (PORT 5000)
echo ====================================================================
echo.
echo Starting Flask + SQLAlchemy Server (mini.py) at http://127.0.0.1:5000...
echo.
start "" http://127.0.0.1:5000
py "%~dp0mini.py"
pause

