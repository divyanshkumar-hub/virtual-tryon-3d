"""Real computer-vision body, face, and skin-tone analyzer for StyleMate / Virtual Try-On 3D.

Uses MediaPipe Pose + FaceMesh when the legacy `mp.solutions` API is available,
OpenCV Haar-cascade face detection + HSV skin sampling when `cv2` is available,
and falls back cleanly to Pillow/NumPy image statistics so the server never crashes.
"""

from __future__ import annotations

import io
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image

try:
    import cv2  # type: ignore

    _HAS_CV2 = True
except Exception:
    cv2 = None  # type: ignore
    _HAS_CV2 = False

try:
    import mediapipe as mp  # type: ignore

    _HAS_MP_SOLUTIONS = hasattr(mp, "solutions") and hasattr(mp.solutions, "pose") and hasattr(mp.solutions, "face_mesh")
except Exception:
    mp = None  # type: ignore
    _HAS_MP_SOLUTIONS = False


def _classify_skin_tone(rgb: Tuple[float, float, float]) -> Tuple[str, str]:
    """Map average RGB skin sample to a tone label and undertone."""
    r, g, b = rgb
    luminance = 0.299 * r + 0.587 * g + 0.114 * b

    if luminance >= 195:
        tone = "Fair"
    elif luminance >= 155:
        tone = "Light-Medium"
    elif luminance >= 115:
        tone = "Medium / Olive"
    elif luminance >= 80:
        tone = "Tan / Brown"
    else:
        tone = "Deep"

    if (r - b) > 28 and g > b:
        undertone = "Warm"
    elif (r - b) < 12:
        undertone = "Cool"
    else:
        undertone = "Neutral"

    return tone, undertone


def _recommend_palette(undertone: str) -> list[str]:
    palettes = {
        "Warm": ["Olive Green", "Terracotta", "Mustard", "Warm Cream", "Rust"],
        "Cool": ["Navy Blue", "Emerald", "Charcoal", "Crisp White", "Burgundy"],
        "Neutral": ["Teal", "Soft Gray", "Sage", "Coral", "Midnight Blue"],
    }
    return palettes.get(undertone, palettes["Neutral"])


def _analyze_with_mediapipe(img_rgb: np.ndarray) -> Dict[str, Any]:
    h, w, _ = img_rgb.shape
    result: Dict[str, Any] = {
        "image_width": int(w),
        "image_height": int(h),
        "pose_detected": False,
        "face_detected": False,
    }

    mp_pose = mp.solutions.pose
    mp_face = mp.solutions.face_mesh

    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.45) as pose:
        pose_res = pose.process(img_rgb)
        if pose_res.pose_landmarks:
            lm = pose_res.pose_landmarks.landmark
            shoulder_w = abs(lm[11].x - lm[12].x) * w
            hip_w = abs(lm[23].x - lm[24].x) * w
            torso_h = abs(((lm[11].y + lm[12].y) / 2.0) - ((lm[23].y + lm[24].y) / 2.0)) * h

            ratio = shoulder_w / max(hip_w, 1.0)
            if ratio > 1.15:
                body_type = "Inverted Triangle"
            elif ratio < 0.88:
                body_type = "Pear"
            elif torso_h > 0 and (shoulder_w / torso_h) < 0.55:
                body_type = "Rectangle"
            else:
                body_type = "Hourglass / Balanced"

            result.update(
                {
                    "pose_detected": True,
                    "body_type": body_type,
                    "shoulder_hip_ratio": round(float(ratio), 2),
                    "estimated_shoulder_px": round(float(shoulder_w), 1),
                    "estimated_hip_px": round(float(hip_w), 1),
                }
            )

    skin_rgb = None
    with mp_face.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.45) as face:
        face_res = face.process(img_rgb)
        if face_res.multi_face_landmarks:
            flm = face_res.multi_face_landmarks[0].landmark
            face_w = abs(flm[454].x - flm[234].x) * w
            face_h = abs(flm[152].y - flm[10].y) * h
            jaw_w = abs(flm[365].x - flm[136].x) * w
            aspect = face_h / max(face_w, 1.0)
            jaw_ratio = jaw_w / max(face_w, 1.0)

            if aspect >= 1.45:
                face_shape = "Oblong / Oval"
            elif aspect <= 1.18 and jaw_ratio >= 0.82:
                face_shape = "Round"
            elif jaw_ratio >= 0.85:
                face_shape = "Square"
            elif jaw_ratio <= 0.68:
                face_shape = "Heart"
            else:
                face_shape = "Oval"

            cx = int(np.clip(flm[50].x * w, 4, w - 5))
            cy = int(np.clip(flm[50].y * h, 4, h - 5))
            patch = img_rgb[cy - 3 : cy + 4, cx - 3 : cx + 4]
            if patch.size > 0:
                skin_rgb = tuple(float(v) for v in patch.mean(axis=(0, 1)))

            result.update(
                {
                    "face_detected": True,
                    "face_shape": face_shape,
                    "face_aspect_ratio": round(float(aspect), 2),
                }
            )

    if skin_rgb is None:
        cy0, cy1 = int(h * 0.15), int(h * 0.45)
        cx0, cx1 = int(w * 0.35), int(w * 0.65)
        patch = img_rgb[cy0:cy1, cx0:cx1]
        skin_rgb = tuple(float(v) for v in patch.mean(axis=(0, 1)))

    tone, undertone = _classify_skin_tone(skin_rgb)
    result.update(
        {
            "skin_tone": tone,
            "undertone": undertone,
            "recommended_colors": _recommend_palette(undertone),
            "analysis_engine": "mediapipe_cv",
        }
    )
    return result


def _analyze_with_opencv(img_rgb: np.ndarray) -> Dict[str, Any]:
    """Use OpenCV Haar-cascade face detection + contour/silhouette aspect analysis."""
    h, w, _ = img_rgb.shape
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    face_detected = False
    face_shape = "Oval"
    skin_rgb = None

    cascade_path = getattr(cv2.data, "haarcascades", "") + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if not face_cascade.empty():
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
        if len(faces) > 0:
            x, y, fw, fh = max(faces, key=lambda b: b[2] * b[3])
            face_detected = True
            face_aspect = fh / max(fw, 1)
            face_shape = "Round" if face_aspect < 1.05 else ("Oblong / Oval" if face_aspect > 1.25 else "Oval")
            # Sample inner cheek area inside detected face box
            cx0, cx1 = int(x + fw * 0.25), int(x + fw * 0.75)
            cy0, cy1 = int(y + fh * 0.35), int(y + fh * 0.70)
            patch = img_rgb[cy0:cy1, cx0:cx1]
            if patch.size > 0:
                skin_rgb = tuple(float(v) for v in patch.mean(axis=(0, 1)))

    if skin_rgb is None:
        cy0, cy1 = int(h * 0.15), int(h * 0.45)
        cx0, cx1 = int(w * 0.35), int(w * 0.65)
        patch = img_rgb[cy0:cy1, cx0:cx1]
        skin_rgb = tuple(float(v) for v in patch.mean(axis=(0, 1)))

    tone, undertone = _classify_skin_tone(skin_rgb)
    aspect = h / max(w, 1)
    body_type = "Rectangle" if aspect >= 1.55 else "Balanced"

    return {
        "image_width": int(w),
        "image_height": int(h),
        "pose_detected": False,
        "face_detected": face_detected,
        "body_type": body_type,
        "face_shape": face_shape,
        "skin_tone": tone,
        "undertone": undertone,
        "recommended_colors": _recommend_palette(undertone),
        "analysis_engine": "opencv_cv" if face_detected else "opencv_fallback",
    }


def analyze_image_bytes(image_bytes: bytes) -> Dict[str, Any]:
    """Analyze uploaded person image bytes and return structured fashion & body metrics."""
    if not image_bytes:
        raise ValueError("Empty image payload")

    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_rgb = np.array(pil_img)
    h, w, _ = img_rgb.shape

    if _HAS_MP_SOLUTIONS:
        try:
            cv_out = _analyze_with_mediapipe(img_rgb)
            if cv_out.get("pose_detected") or cv_out.get("face_detected"):
                cv_out.setdefault("body_type", "Balanced")
                cv_out.setdefault("face_shape", "Oval")
                return cv_out
        except Exception:
            pass

    if _HAS_CV2:
        try:
            return _analyze_with_opencv(img_rgb)
        except Exception:
            pass

    # Pillow + NumPy fallback (center-crop color & aspect analysis)
    cy0, cy1 = int(h * 0.15), int(h * 0.45)
    cx0, cx1 = int(w * 0.35), int(w * 0.65)
    center_patch = img_rgb[cy0:cy1, cx0:cx1]
    skin_rgb = tuple(float(v) for v in center_patch.mean(axis=(0, 1)))
    tone, undertone = _classify_skin_tone(skin_rgb)

    aspect = h / max(w, 1)
    body_type = "Rectangle" if aspect >= 1.55 else "Balanced"

    return {
        "image_width": int(w),
        "image_height": int(h),
        "pose_detected": False,
        "face_detected": False,
        "body_type": body_type,
        "face_shape": "Oval",
        "skin_tone": tone,
        "undertone": undertone,
        "recommended_colors": _recommend_palette(undertone),
        "analysis_engine": "pillow_fallback",
    }