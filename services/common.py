"""Shared helpers for the experimental model services (ports 5101-5103)."""
from __future__ import annotations

import io
import time
from contextlib import contextmanager

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps

try:
    import torch

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:  # height service can run without torch
    torch = None
    DEVICE = "cpu"


def make_app(title: str) -> FastAPI:
    app = FastAPI(title=title)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    return app


async def read_image(file: UploadFile) -> Image.Image:
    data = await file.read()
    if not data:
        raise HTTPException(400, "empty upload")
    try:
        img = Image.open(io.BytesIO(data))
        f35 = None
        try:
            f35 = img.getexif().get_ifd(0x8769).get(0xA405)  # FocalLengthIn35mmFilm
        except Exception:  # noqa: BLE001
            pass
        img = ImageOps.exif_transpose(img).convert("RGB")  # respect phone rotation
        img.info["focal_35mm"] = float(f35) if f35 else None
        return img
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"not an image: {exc}") from exc


@contextmanager
def timer(out: dict, key: str = "latency_ms"):
    t0 = time.perf_counter()
    yield
    out[key] = round((time.perf_counter() - t0) * 1000, 1)
