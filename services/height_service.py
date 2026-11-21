"""Port 5102 - height estimation experiments.

Pixel height of the person comes from the MediaPipe pose *segmentation mask*
(top of hair -> sole) instead of landmark heuristics. That pixel height is
turned into centimetres three independent ways:

  1. user_height   - what the app does today (scale = user height / px height)
  2. aruco         - a printed ArUco marker (DICT_4X4_50, known side length)
                     held at the body plane gives cm-per-pixel directly
  3. metric_depth  - Depth-Anything-V2 metric depth + camera focal length
                     (EXIF if present, otherwise a 26 mm phone-lens prior):
                     H = px_height * Z / f_px

Run:  services/.venv/Scripts/python -m uvicorn height_service:app --port 5102 --app-dir services
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from fastapi import File, Form, HTTPException, UploadFile
from PIL import Image

from common import DEVICE, make_app, read_image, timer

ROOT = Path(__file__).resolve().parent
POSE_MODEL = os.environ.get("POSE_MODEL", str(ROOT / "models" / "pose_landmarker_heavy.task"))
DEPTH_MODEL = os.environ.get("DEPTH_MODEL", "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf")

app = make_app("Height service")

_pose = mp.tasks.vision.PoseLandmarker.create_from_options(
    mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=POSE_MODEL),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        output_segmentation_masks=True,
    )
)
_aruco = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
                                 cv2.aruco.DetectorParameters())
_depth = None  # lazy - only loaded when metric depth is requested


def _depth_pipe():
    global _depth
    if _depth is None:
        from transformers import pipeline

        _depth = pipeline("depth-estimation", model=DEPTH_MODEL, device=0 if DEVICE == "cuda" else -1)
    return _depth


def person_extent(rgb: np.ndarray):
    """Returns (top_y, bottom_y, mask, landmarks) in pixels, or raises."""
    res = _pose.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
    if not res.pose_landmarks:
        raise ValueError("no person detected")
    h, w = rgb.shape[:2]
    lm = res.pose_landmarks[0]
    mask = res.segmentation_masks[0].numpy_view().squeeze() > 0.5
    # keep only the connected blob that contains the hips (drops other people / noise)
    n, labels = cv2.connectedComponents(mask.astype(np.uint8))
    hx, hy = int((lm[23].x + lm[24].x) / 2 * w), int((lm[23].y + lm[24].y) / 2 * h)
    hx, hy = min(max(hx, 0), w - 1), min(max(hy, 0), h - 1)
    if n > 1 and labels[hy, hx] > 0:
        mask = labels == labels[hy, hx]
    rows = np.where(mask.any(axis=1))[0]
    if rows.size == 0:
        raise ValueError("empty segmentation")
    top, bottom = int(rows[0]), int(rows[-1])
    vis = lambda i: getattr(lm[i], "visibility", 1.0)  # noqa: E731
    feet_visible = max(vis(29), vis(30), vis(31), vis(32)) > 0.5 and bottom < h - 2
    head_visible = vis(0) > 0.5 and top > 1
    return top, bottom, mask, lm, feet_visible and head_visible


def focal_px(img: Image.Image, w: int, h: int, f35_default: float = 26.0):
    f35 = img.info.get("focal_35mm")
    src = "exif" if f35 else "prior_26mm"
    f35 = f35 or f35_default
    return f35 * math.hypot(w, h) / math.hypot(36, 24), src


@app.get("/health")
def health():
    return {"ok": True, "pose_model": Path(POSE_MODEL).name, "depth_model": DEPTH_MODEL, "device": DEVICE}


@app.post("/estimate")
async def estimate(file: UploadFile = File(...),
                   user_height_cm: float | None = Form(None),
                   marker_cm: float = Form(15.0),
                   use_depth: bool = Form(True),
                   focal_35mm: float | None = Form(None)):
    pil = await read_image(file)
    raw = pil
    out: dict = {"methods": {}}
    with timer(out):
        rgb = np.asarray(pil)
        if max(rgb.shape[:2]) > 1600:
            s = 1600 / max(rgb.shape[:2])
            rgb = cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        h, w = rgb.shape[:2]
        try:
            top, bottom, mask, lm, full_body = person_extent(rgb)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        px_h = bottom - top
        out.update(pixel_height=px_h, full_body_visible=full_body, image_size=[w, h])

        if user_height_cm:
            out["methods"]["user_height"] = {"height_cm": user_height_cm, "cm_per_px": user_height_cm / px_h}

        corners, ids, _ = _aruco.detectMarkers(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
        if ids is not None and len(ids):
            c = corners[0][0]
            side_px = np.mean([np.linalg.norm(c[i] - c[(i + 1) % 4]) for i in range(4)])
            cm_px = marker_cm / side_px
            out["methods"]["aruco"] = {"height_cm": round(px_h * cm_px, 1), "marker_side_px": round(float(side_px), 1),
                                       "cm_per_px": cm_px}

        if use_depth:
            depth = np.asarray(_depth_pipe()(Image.fromarray(rgb))["predicted_depth"].squeeze())
            depth = cv2.resize(depth.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
            z = float(np.median(depth[mask & (depth > 0)]))  # metres to the body
            if focal_35mm:
                f, src = focal_35mm * math.hypot(w, h) / math.hypot(36, 24), "given"
            else:
                f, src = focal_px(raw, w, h)
            out["methods"]["metric_depth"] = {"height_cm": round(px_h * z / f * 100, 1), "distance_m": round(z, 2),
                                              "focal_px": round(f, 1), "focal_source": src}

        best = next((m for m in ("aruco", "user_height", "metric_depth") if m in out["methods"]), None)
        out["best_method"] = best
        if best:
            cm_px = out["methods"][best].get("cm_per_px") or out["methods"][best]["height_cm"] / px_h

            def d(a, b):
                return math.hypot((lm[a].x - lm[b].x) * w, (lm[a].y - lm[b].y) * h) * cm_px

            out["height_cm"] = out["methods"][best]["height_cm"]
            out["measurements_cm"] = {
                "shoulder_width": round(d(11, 12), 1),  # landmark-to-landmark (under-estimates biacromial)
                "hip_width": round(d(23, 24), 1),
                "inseam_approx": round((d(23, 25) + d(25, 27) + d(24, 26) + d(26, 28)) / 2, 1),
                "arm_length": round((d(11, 13) + d(13, 15) + d(12, 14) + d(14, 16)) / 2, 1),
            }
            # silhouette width at chest / waist / hip rows -> front width in cm
            def width_at(y_norm):
                row = int(min(max(y_norm * h, 0), h - 1))
                cols = np.where(mask[row])[0]
                return round(float(cols[-1] - cols[0]) * cm_px, 1) if cols.size else None

            sh_y = (lm[11].y + lm[12].y) / 2
            hp_y = (lm[23].y + lm[24].y) / 2
            out["silhouette_width_cm"] = {
                "chest": width_at(sh_y + 0.25 * (hp_y - sh_y)),
                "waist": width_at(sh_y + 0.70 * (hp_y - sh_y)),
                "hip": width_at(hp_y),
            }
        for m in out["methods"].values():
            m.pop("cm_per_px", None)
    return json.loads(json.dumps(out, default=float))  # numpy scalars -> plain floats
