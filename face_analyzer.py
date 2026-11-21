"""Face-shape analysis from MediaPipe Face Landmarker (478 landmarks).

Ported from the VisionFit-AI project's ai_vision/face_analyzer.py (rule-based face-shape
classifier and the landmark-ratio definitions) and adapted to:
  * the MediaPipe Tasks API (mp.solutions is no longer available in mediapipe 1.x),
  * full-body photos: the face is cropped from the age service's face box or from pose
    landmarks (nose + shoulders) and upscaled before landmarking,
  * cheek skin sampling for skin tone, and frame advice per face shape.
"""
from __future__ import annotations

import math
import os
import threading
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2  # type: ignore
    import mediapipe as mp  # type: ignore
    _HAS_MP = True
except Exception:  # pragma: no cover - fallback environments
    cv2 = None  # type: ignore
    mp = None  # type: ignore
    _HAS_MP = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FACE_MODEL = os.path.join(BASE_DIR, "face_landmarker.task")

# Face Mesh topology indices (shared by Face Landmarker)
CHEEK_L, CHEEK_R = 234, 454        # face width at the cheekbones
FOREHEAD_TOP, CHIN = 10, 152       # face height
JAW_L, JAW_R = 172, 397            # jaw width
BROW_L, BROW_R = 103, 332          # forehead width
CHEEK_SKIN = (50, 280)             # centre of each cheek, for skin colour

# Frame styles that balance each face shape (standard optician guidance)
FRAME_ADVICE = {
    "Oval": ["Wayfarer", "Aviator", "Square", "Round"],
    "Round": ["Rectangular", "Square", "Wayfarer", "Angular cat-eye"],
    "Square": ["Round", "Oval", "Aviator", "Rimless"],
    "Heart": ["Aviator", "Bottom-heavy", "Oval", "Light rimless"],
    "Diamond": ["Cat-eye", "Oval", "Rimless", "Browline"],
    "Long / Rectangle": ["Oversized", "Square", "Deep wayfarer", "Decorative temples"],
}

_lock = threading.Lock()
_landmarker = None


def _get_landmarker():
    global _landmarker
    if not _HAS_MP or not os.path.exists(FACE_MODEL):
        return None
    with _lock:
        if _landmarker is None:
            opts = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=FACE_MODEL),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_faces=1, min_face_detection_confidence=0.4, min_face_presence_confidence=0.4)
            _landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(opts)
    return _landmarker


# Thresholds calibrated on 240 UTKFace faces measured with these exact landmarks. VisionFit's
# original cut-offs (forehead >= 0.85-0.92, length >= 1.45) are never reached by these
# landmark definitions (forehead/width p90 = 0.76, length p90 = 1.25), which made every face
# Diamond, Round or Oval. The rule structure is unchanged; the cut-offs are population
# percentiles: length p35/p85, jaw p25/p70, forehead p25/p70.
SHORT_FACE, LONG_FACE = 1.128, 1.222
NARROW_JAW, WIDE_JAW = 0.801, 0.830
NARROW_BROW, WIDE_BROW = 0.706, 0.738


def classify_face_shape(width_height: float, jaw_to_width: float, forehead_to_width: float) -> str:
    """Rule-based face shape from three proportions (VisionFit-AI rule structure, recalibrated)."""
    if width_height <= 0:
        return "Unclassified"
    length = 1 / width_height                      # face height / cheek width
    if length >= LONG_FACE:
        return "Long / Rectangle"
    if forehead_to_width >= WIDE_BROW and jaw_to_width <= NARROW_JAW:
        return "Heart"
    if forehead_to_width <= NARROW_BROW and jaw_to_width <= NARROW_JAW:
        return "Diamond"
    if length <= SHORT_FACE and jaw_to_width >= WIDE_JAW:
        return "Square"
    if length <= SHORT_FACE:
        return "Round"
    return "Oval"


def _face_region(rgb: np.ndarray, face_box: Optional[Sequence[int]], pose_landmarks) -> Optional[Tuple[int, int, int, int]]:
    """Square-ish region around the face, from a detector box or from pose (nose + shoulders)."""
    h, w = rgb.shape[:2]
    if face_box:
        x0, y0, x1, y1 = [int(v) for v in face_box]
        pad = int(max(x1 - x0, y1 - y0) * 0.35)
        return max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)
    if pose_landmarks:
        nose, ls, rs = pose_landmarks[0], pose_landmarks[11], pose_landmarks[12]
        if min(getattr(p, "visibility", 1.0) or 0 for p in (nose, ls, rs)) >= 0.5:
            sw = math.hypot((ls.x - rs.x) * w, (ls.y - rs.y) * h)
            if sw >= 1:
                nx, ny = int(nose.x * w), int(nose.y * h)
                return (max(0, nx - int(sw * 0.45)), max(0, int(ny - sw * 0.6)),
                        min(w, nx + int(sw * 0.45)), min(h, int(ny + sw * 0.45)))
    return None


def _detect(landmarker, rgb: np.ndarray):
    res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
    return res.face_landmarks[0] if res.face_landmarks else None


def analyze_face(rgb: np.ndarray, face_box: Optional[Sequence[int]] = None, pose_landmarks=None) -> Dict[str, Any]:
    """rgb: HxWx3 uint8 image. Returns face shape, proportions, cheek colour and frame advice."""
    landmarker = _get_landmarker()
    if landmarker is None:
        return {"face_detected": False, "face_shape": "Unavailable", "reason": "face model not available"}

    crop, crop_used = rgb, False
    region = _face_region(rgb, face_box, pose_landmarks)
    if region and region[2] > region[0] and region[3] > region[1]:
        x0, y0, x1, y1 = region
        crop = rgb[y0:y1, x0:x1]
        s = max(1.0, 512 / max(crop.shape[:2]))      # small faces in full-body photos need upscaling
        crop = cv2.resize(crop, (int(crop.shape[1] * s), int(crop.shape[0] * s)), interpolation=cv2.INTER_CUBIC)
        crop_used = True
    with _lock:
        lm = _detect(landmarker, crop)
        if lm is None and crop_used:
            crop, crop_used = rgb, False
            lm = _detect(landmarker, crop)
    if lm is None:
        return {"face_detected": False, "face_shape": "Not detected",
                "reason": "No clear front-facing face found. Use a photo with the face visible."}

    fh, fw = crop.shape[:2]
    pt = lambda i: (lm[i].x * fw, lm[i].y * fh)  # noqa: E731
    dist = lambda a, b: math.dist(pt(a), pt(b))  # noqa: E731
    cheek, height, jaw, brow = dist(CHEEK_L, CHEEK_R), dist(FOREHEAD_TOP, CHIN), dist(JAW_L, JAW_R), dist(BROW_L, BROW_R)
    if cheek <= 0 or height <= 0:
        return {"face_detected": False, "face_shape": "Unclassified", "reason": "could not measure the face"}

    ratios = {"face_width_height_ratio": round(cheek / height, 4),
              "jaw_to_face_width_ratio": round(jaw / cheek, 4),
              "forehead_to_face_width_ratio": round(brow / cheek, 4)}
    shape = classify_face_shape(*ratios.values())
    # yaw check: a turned head makes the ratios unreliable
    nose_x = lm[1].x * fw
    yaw = abs((nose_x - pt(CHEEK_L)[0]) / max(1e-6, pt(CHEEK_R)[0] - pt(CHEEK_L)[0]) - 0.5)

    cheek_rgb = None
    r = max(2, int(cheek * 0.04))
    patches = []
    for i in CHEEK_SKIN:
        cx, cy = (int(v) for v in pt(i))
        p = crop[max(0, cy - r):cy + r, max(0, cx - r):cx + r]
        if p.size:
            patches.append(p.reshape(-1, 3))
    if patches:
        cheek_rgb = tuple(float(v) for v in np.median(np.concatenate(patches), axis=0))

    return {
        "face_detected": True,
        "face_shape": shape,
        "face_shape_confidence": "low" if yaw > 0.12 else "medium",
        "head_turn": round(float(yaw), 3),
        "ratios": ratios,
        "frame_advice": FRAME_ADVICE.get(shape, []),
        "cheek_rgb": cheek_rgb,
        "face_crop_used": crop_used,
        "landmark_count": len(lm),
    }
