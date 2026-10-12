@echo off
title V-STYLE 3D - AI Virtual Try-On Studio
color 0b
echo ========================================================
echo        V-STYLE 3D: AI VIRTUAL TRY-ON STUDIO
echo                  Phase 1 Evaluation
echo ========================================================
echo.
echo Starting the 3D Virtual Try-On Studio...
echo.

:: Check for py or python launcher
where py >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [✓] Python Launcher detected (py)
    py "%~dp0server.py"
    goto end
)

where python >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [✓] Python detected
    python "%~dp0server.py"
    goto end
)

echo [!] Python was not found in PATH.
echo [!] Opening web application directly in your default browser...
start "" "%~dp0web_app\index.html"

:end
pause

