"""Port 5100 - browser test bench + process manager for the model services.

  services/.venv/Scripts/python -m uvicorn bench_service:app --port 5100 --app-dir services
  then open http://localhost:5100
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
PY = sys.executable
(ROOT / "logs").mkdir(exist_ok=True)

SERVICES = {
    "age": {"port": 5101, "module": "age_service", "gpu_gb": 0.5},
    "height": {"port": 5102, "module": "height_service", "gpu_gb": 0.3},
    "fashion": {"port": 5103, "module": "fashion_service", "gpu_gb": 0.4},
}

app = FastAPI(title="Model test bench")
for mount, path in {
    "/data/fashion": ROOT / "eval_data" / "fashion",
    "/data/utkface": ROOT / "eval_data" / "utkface",
    "/data/height": ROOT / "eval_results" / "height",
    "/samples": PROJECT / "samples",
}.items():
    if path.exists():
        app.mount(mount, StaticFiles(directory=path), name=mount.strip("/").replace("/", "_"))


@app.get("/")
def index():
    return FileResponse(ROOT / "bench.html")


def _pid_on(port: int) -> int | None:
    out = subprocess.run(["netstat", "-ano", "-p", "TCP"], capture_output=True, text=True).stdout
    m = re.search(rf"127\.0\.0\.1:{port}\s+\S+\s+LISTENING\s+(\d+)|0\.0\.0\.0:{port}\s+\S+\s+LISTENING\s+(\d+)", out)
    return int(m.group(1) or m.group(2)) if m else None


@app.get("/api/status")
def status():
    res = {}
    for name, s in SERVICES.items():
        try:
            h = requests.get(f"http://127.0.0.1:{s['port']}/health", timeout=1.5).json()
            res[name] = {"state": "up", **s, "health": h}
        except Exception:  # noqa: BLE001
            res[name] = {"state": "starting" if _pid_on(s["port"]) else "down", **s}
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=5).stdout.strip().split(",")
        res["_gpu"] = {"used_mb": int(q[0]), "total_mb": int(q[1])}
    except Exception:  # noqa: BLE001
        pass
    return res


@app.post("/api/start/{name}")
def start(name: str):
    if name not in SERVICES:
        raise HTTPException(404)
    s = SERVICES[name]
    if _pid_on(s["port"]):
        return {"ok": True, "note": "already running"}
    log = open(ROOT / "logs" / f"{name}.log", "w")
    subprocess.Popen([PY, "-m", "uvicorn", f"{s['module']}:app", "--port", str(s["port"])], cwd=ROOT,
                     stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONUNBUFFERED": "1"},
                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return {"ok": True}


@app.post("/api/stop/{name}")
def stop(name: str):
    if name not in SERVICES:
        raise HTTPException(404)
    pid = _pid_on(SERVICES[name]["port"])
    if pid:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)
    return {"ok": True, "stopped_pid": pid}


@app.post("/api/results")
async def save_results(payload: dict):
    """The page posts each autonomous run here so it can be reviewed later."""
    import json
    import time

    d = ROOT / "eval_results" / "bench_runs"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"run_{time.strftime('%Y%m%d_%H%M%S')}.json"
    p.write_text(json.dumps(payload, indent=1))
    return {"saved": p.name}


@app.get("/api/log/{name}")
def log(name: str):
    p = ROOT / "logs" / f"{name}.log"
    if name not in SERVICES or not p.exists():
        raise HTTPException(404)
    lines = [ln for ln in p.read_text(errors="replace").splitlines()
             if not ln.startswith(("INFO:", "WARNING: [model_load", "W0000", "I0000"))]
    return {"tail": lines[-25:]}
