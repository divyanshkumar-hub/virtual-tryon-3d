@echo off
REM Starts the AI model services, each in its own window.
REM   5101 age (MiVOLO v2)   5102 height   5103 FashionCLIP   5104 cloth analysis (VisionFit)   5100 test bench
REM Options (any order):
REM   eval   - also start the OpenAI-CLIP baseline on 5113 (only used by eval/run_evals.py)
REM   bench  - open the test bench in the browser
setlocal
pushd "%~dp0"
set PY=%~dp0.venv\Scripts\python.exe
start "age:5101"      "%PY%" -m uvicorn age_service:app --port 5101
start "height:5102"   "%PY%" -m uvicorn height_service:app --port 5102
start "fashion:5103"  "%PY%" -m uvicorn fashion_service:app --port 5103
start "cloth:5104"    "%PY%" -m uvicorn cloth_service:app --port 5104
start "bench:5100"    "%PY%" -m uvicorn bench_service:app --port 5100
for %%A in (%*) do (
  if /I "%%A"=="eval"  start "clip-base:5113" cmd /c "set FASHION_CLIP_MODEL=openai/clip-vit-base-patch32&& "%PY%" -m uvicorn fashion_service:app --port 5113"
  if /I "%%A"=="bench" start "" http://localhost:5100
)
popd
endlocal
