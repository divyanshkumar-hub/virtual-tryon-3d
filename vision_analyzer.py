"""vision_analyzer.py - Real Computer Vision Body Measurement & Fashion Analysis.

Extracts real 33-point body pose landmarks using MediaPipe Pose Landmarker (.task),
computes anthropometric measurements (shoulder width, chest, waist, and hip circumference),
classifies body type, face shape, skin tone and undertone, and recommends matching
clothing, shoes, sunglasses, and color palettes.
Includes robust OpenCV and Pillow/NumPy fallbacks so it never crashes.
"""

from __future__ import annotations

import io
import math
import os
import struct
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps

from face_analyzer import analyze_face

try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except Exception:
    cv2 = None  # type: ignore
    _HAS_CV2 = False

try:
    import mediapipe as mp  # type: ignore
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    _HAS_MEDIAPIPE = True
except Exception:
    mp = None  # type: ignore
    _HAS_MEDIAPIPE = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Candidate paths for pose_landmarker_lite.task model
CANDIDATE_MODEL_PATHS = [
    os.path.join(BASE_DIR, "pose_landmarker_lite.task"),
    r"C:\code\stylemate-v2\pose_landmarker_lite.task",
    r"C:\code\virtual_tryon_3d\pose_landmarker_lite.task",
    r"C:\Users\Divyansh Kumar\OneDrive\Desktop\mini project\app.py\app.py\pose_landmarker_lite.task",
]

MODEL_PATH: Optional[str] = None
for path in CANDIDATE_MODEL_PATHS:
    if os.path.exists(path):
        MODEL_PATH = path
        break

# MediaPipe Pose Landmark Indices (BlazePose 33 landmarks)
NOSE = 0
L_EYE = 2
R_EYE = 5
L_EAR = 7
R_EAR = 8
MOUTH_L = 9
MOUTH_R = 10
L_SHOULDER = 11
R_SHOULDER = 12
L_ELBOW = 13
R_ELBOW = 14
L_WRIST = 15
R_WRIST = 16
L_HIP = 23
R_HIP = 24
L_KNEE = 25
R_KNEE = 26
L_ANKLE = 27
R_ANKLE = 28
L_HEEL = 29
R_HEEL = 30
L_FOOT_INDEX = 31
R_FOOT_INDEX = 32

_detector = None
_detector_lock = threading.Lock()
_infer_lock = threading.Lock()


def get_detector():
    """Lazily load and reuse the MediaPipe PoseLandmarker model."""
    global _detector
    if not _HAS_MEDIAPIPE:
        return None
    if _detector is None:
        with _detector_lock:
            if _detector is None:
                if not MODEL_PATH or not os.path.exists(MODEL_PATH):
                    return None
                try:
                    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
                    options = mp_vision.PoseLandmarkerOptions(
                        base_options=base_options,
                        min_pose_detection_confidence=0.15,
                        min_pose_presence_confidence=0.15,
                        min_tracking_confidence=0.15,
                    )
                    _detector = mp_vision.PoseLandmarker.create_from_options(options)
                except Exception:
                    return None
    return _detector


def _pixel_dist(a, b, img_w: float, img_h: float) -> float:
    return math.hypot((a.x - b.x) * img_w, (a.y - b.y) * img_h)


def _circumference_from_width(width_cm: float, ratio: float = 0.70) -> float:
    """Ramanujan approximation for ellipse circumference from major width & depth ratio."""
    a = width_cm / 2.0
    b = (width_cm * ratio) / 2.0
    return math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))


def classify_body_type(shoulder_cm: float, waist_cm: float, hip_cm: float) -> str:
    """
    Classify body type using shoulder width (cm), waist circumference (cm), and hip circumference (cm).
    """
    waist_w = waist_cm / 2.7
    hip_w = hip_cm / 2.8
    wh_ratio = waist_cm / max(hip_cm, 1.0)
    sh_w_ratio = shoulder_cm / max(waist_w, 1.0)
    sh_hip_diff = shoulder_cm - hip_w

    if sh_w_ratio > 1.35 and sh_hip_diff > 4.0:
        return "Inverted Triangle (V-Shape Athletic)"
    elif abs(sh_hip_diff) <= 4.0 and wh_ratio < 0.82:
        return "Hourglass / Balanced Proportions"
    elif hip_w > shoulder_cm + 2.0:
        return "Pear / Hip-Dominant Build"
    elif wh_ratio > 0.86:
        return "Rectangle / Athletic / Lean Build"
    else:
        return "Oval / Rounded Proportions"


def get_clothing_size(chest_cm: float, gender: str = "unisex") -> str:
    """Accurate apparel size classification based on chest circumference."""
    is_female = str(gender).lower() in ["female", "woman", "f"]
    if is_female:
        if chest_cm < 82:
            return "XS"
        elif chest_cm < 89:
            return "S"
        elif chest_cm < 97:
            return "M"
        elif chest_cm < 105:
            return "L"
        elif chest_cm < 113:
            return "XL"
        return "XXL"
    else:
        if chest_cm < 88:
            return "XS"
        elif chest_cm < 95:
            return "S"
        elif chest_cm < 103:
            return "M"
        elif chest_cm < 111:
            return "L"
        elif chest_cm < 119:
            return "XL"
        return "XXL"


KIDS_SIZES = [  # (max height cm, label) - typical EU/UK kidswear by stature
    (86, "12-18M"), (92, "18-24M"), (98, "2-3Y"), (104, "3-4Y"), (110, "4-5Y"), (116, "5-6Y"),
    (122, "6-7Y"), (128, "7-8Y"), (134, "8-9Y"), (140, "9-10Y"), (146, "10-11Y"), (152, "11-12Y"),
    (158, "12-13Y"), (164, "13-14Y"),
]


def get_kids_size(height_cm: float) -> str:
    for max_h, label in KIDS_SIZES:
        if height_cm <= max_h:
            return label
    return "14Y+"


def is_child(age: Optional[float]) -> bool:
    return age is not None and age < 14


def measurement_confidence(method: str, measurements: Dict[str, Any], height_source: str = "user") -> Dict[str, Any]:
    """Honest 0-100 confidence for the body measurements, with the reasons behind it."""
    reasons = []
    if method == "mediapipe_real":
        vis = float(measurements.get("pose_visibility", 0.6))
        score = 45 + 35 * vis
        reasons.append(f"body landmarks found (visibility {vis:.0%})")
        if measurements.get("full_body_visible"):
            score += 10
            reasons.append("whole body in frame")
        else:
            score -= 15
            reasons.append("head or feet cut off - height scale is extrapolated")
    elif method == "opencv_contour":
        score = 30
        reasons.append("no landmarks; rough silhouette only")
    else:
        score = 10
        reasons.append("no person detected; values are population averages")
    if height_source == "marker":
        score += 10
        reasons.append("scale from printed marker")
    elif height_source == "estimated":
        score -= 20
        reasons.append("height not given - estimated from age/gender")
    score = int(max(5, min(95, score)))
    level = "high" if score >= 75 else "medium" if score >= 50 else "low"
    return {"score": score, "level": level, "reasons": reasons}


def get_recommendations(body_type: str, size: str) -> Tuple[List[str], List[str], List[str], str]:
    bt = body_type.lower()
    if "child" in bt:
        clothes = [
            f"Soft Cotton Crew T-Shirt (Size {size})",
            f"Pull-On Stretch Jeans (Size {size})",
            f"Zip-Up Fleece Hoodie (Size {size})",
            f"Easy-Fit Cargo Shorts (Size {size})",
        ]
        shoes = ["Velcro Canvas Sneakers", "Lightweight Running Shoes", "Summer Sandals"]
        glasses = ["UV400 Kids Sunglasses", "Flexible Round Frames", "Sports Wrap Shades"]
        face = "Round / Youthful"
    elif "inverted" in bt or "athletic" in bt:
        clothes = [
            f"Navy Tailored Double-Breasted Blazer (Size {size})",
            f"Slim-Fit White Oxford Shirt (Size {size})",
            f"Relaxed Straight-Leg Denim Jeans (Size {size})",
            f"Cyber Urban Hoodie (Size {size})",
        ]
        shoes = ["Classic White Leather Sneakers", "Suede Chelsea Boots", "Chunky Streetwear Runners"]
        glasses = ["Gold Aviator Sunglasses", "Tortoiseshell Wayfarer Frames", "Vintage Clubmaster Shades"]
        face = "Oval / Defined"
    elif "hourglass" in bt or "balanced" in bt:
        clothes = [
            f"Fitted Stretch Chino Trousers (Size {size})",
            f"Slim V-Neck Knit Sweater (Size {size})",
            f"Tailored Wool Blazer (Size {size})",
            f"White Linen Button-Down Shirt (Size {size})",
        ]
        shoes = ["Brown Leather Derby Shoes", "White Canvas Low-Tops", "Cognac Leather Loafers"]
        glasses = ["Round Retro Metal Sunglasses", "Gold Aviator Sunglasses", "Polarized Navigator Frames"]
        face = "Square / Defined Jawline"
    elif "pear" in bt or "hip" in bt:
        clothes = [
            f"Structured Shoulder Blazer (Size {size})",
            f"Boat-Neck Striped Breton Top (Size {size})",
            f"Dark Wash Straight-Leg Jeans (Size {size})",
            f"Wide-Leg Linen Trousers (Size {size})",
        ]
        shoes = ["Minimalist White Low-Top Sneakers", "Brown Leather Penny Loafers", "High-Top Canvas Trainers"]
        glasses = ["Cat-Eye Acetate Frames", "Angular Wayfarer Sunglasses", "Oversized Square Frames"]
        face = "Heart / Rounded"
    elif "rectangle" in bt or "lean" in bt:
        clothes = [
            f"Textured Layering Hoodie (Size {size})",
            f"Cargo Utility Trousers (Size {size})",
            f"Vintage Leather Rider Jacket (Size {size})",
            f"Relaxed Fit Graphic Tee (Size {size})",
        ]
        shoes = ["High-Top Canvas Trainers", "Retro Basketball Sneakers", "Suede Desert Boots"]
        glasses = ["Hexagonal Metal Frames", "Matte Black Wayfarer", "Slim Rectangular Frames"]
        face = "Oblong / Angular"
    else:
        clothes = [
            f"Relaxed-Fit Linen Shirt (Size {size})",
            f"Straight-Leg Comfort Trousers (Size {size})",
            f"Zip-Up Track Jacket (Size {size})",
            f"Classic Crewneck Sweatshirt (Size {size})",
        ]
        shoes = ["White Leather Sneakers", "Slip-On Canvas Shoes", "Classic Loafers"]
        glasses = ["Round Tortoiseshell Frames", "Gold Aviator Sunglasses", "Gradient Lens Wayfarers"]
        face = "Oval / Rounded"
    return clothes, shoes, glasses, face


def _classify_skin_tone(rgb: Tuple[float, float, float]) -> Tuple[str, str]:
    """Map average RGB sample to a tone label and undertone."""
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


def _recommend_palette(undertone: str) -> List[str]:
    palettes = {
        "Warm": ["Olive Green", "Terracotta", "Mustard", "Warm Cream", "Rust"],
        "Cool": ["Navy Blue", "Emerald", "Charcoal", "Crisp White", "Burgundy"],
        "Neutral": ["Teal", "Soft Gray", "Sage", "Coral", "Midnight Blue"],
    }
    return palettes.get(undertone, palettes["Neutral"])


def run_mediapipe_extraction(
    image_bytes: bytes,
    user_height_cm: float = 175.0,
    gender: str = "unisex"
) -> Dict[str, Any]:
    detector = get_detector()
    if detector is None:
        raise ValueError("MediaPipe detector unavailable")

    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes")

    img_h, img_w = img.shape[:2]

    # Pre-scale image if oversized (e.g. 4K/high-res photos) to maintain detection stability
    max_dim = 1280
    if max(img_h, img_w) > max_dim:
        scale_down = max_dim / float(max(img_h, img_w))
        new_w = int(img_w * scale_down)
        new_h = int(img_h * scale_down)
        img_infer = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        img_infer = img

    img_rgb = cv2.cvtColor(img_infer, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    with _infer_lock:
        results = detector.detect(mp_img)

    if not results.pose_landmarks:
        raise ValueError("No pose landmarks detected")

    lm = results.pose_landmarks[0]

    # Normalize coordinates to original image pixel space
    nose_y = lm[NOSE].y * img_h
    l_shoulder_y = lm[L_SHOULDER].y * img_h
    r_shoulder_y = lm[R_SHOULDER].y * img_h
    shoulder_y = (l_shoulder_y + r_shoulder_y) / 2.0

    l_hip_y = lm[L_HIP].y * img_h
    r_hip_y = lm[R_HIP].y * img_h
    hip_y = (l_hip_y + r_hip_y) / 2.0

    l_knee_y = lm[L_KNEE].y * img_h
    r_knee_y = lm[R_KNEE].y * img_h
    knee_y = (l_knee_y + r_knee_y) / 2.0

    l_ankle_y = lm[L_ANKLE].y * img_h
    r_ankle_y = lm[R_ANKLE].y * img_h
    ankle_y = max(l_ankle_y, r_ankle_y)

    # Foot points (heels & toes)
    foot_points = [
        lm[idx].y * img_h
        for idx in [L_ANKLE, R_ANKLE, L_HEEL, R_HEEL, L_FOOT_INDEX, R_FOOT_INDEX]
        if idx < len(lm) and getattr(lm[idx], "visibility", 1.0) > 0.25
    ]

    # Calculate True Crown (top of skull):
    # In adult humans, distance from nose to top of head is approx 0.85-0.95 of (shoulder_y - nose_y)
    head_offset = max(20.0, (shoulder_y - nose_y) * 0.90) if shoulder_y > nose_y else 50.0
    crown_y = max(0.0, nose_y - head_offset)

    # Calculate True Ground (sole of foot):
    if foot_points:
        ground_y = max(foot_points) + max(4.0, (ankle_y - knee_y) * 0.04 if knee_y < ankle_y else 10.0)
    else:
        ground_y = ankle_y + max(15.0, (ankle_y - knee_y) * 0.12 if knee_y < ankle_y else 25.0)

    # Full standing height vs partial crop fallback
    is_full_body = (ankle_y - nose_y) > img_h * 0.40 and (len(foot_points) > 0 or ankle_y > img_h * 0.70)
    if is_full_body:
        full_pixel_height = ground_y - crown_y
    else:
        if knee_y - crown_y > img_h * 0.35:
            full_pixel_height = (knee_y - crown_y) / 0.74
        elif hip_y - crown_y > img_h * 0.20:
            full_pixel_height = (hip_y - crown_y) / 0.52
        else:
            full_pixel_height = img_h * 0.90

    if full_pixel_height <= 0:
        full_pixel_height = img_h * 0.90

    scale = user_height_cm / full_pixel_height

    # Bounds are proportional to stature (not fixed adult cm) so children and
    # petite/tall users are not forced into an adult range.
    # Biacromial width is ~20-27% of stature, bi-iliac/hip width ~13-22%.
    sh_px = _pixel_dist(lm[L_SHOULDER], lm[R_SHOULDER], img_w, img_h)
    shoulder_cm = round(max(user_height_cm * 0.17, min(user_height_cm * 0.30, sh_px * scale)), 1)

    hip_px = _pixel_dist(lm[L_HIP], lm[R_HIP], img_w, img_h)
    hip_w_cm = round(max(user_height_cm * 0.10, min(user_height_cm * 0.28, hip_px * scale)), 1)

    key_ids = [NOSE, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE]
    visibility = float(np.mean([getattr(lm[i], "visibility", 0.5) or 0.0 for i in key_ids]))

    s_mid_x = (lm[L_SHOULDER].x + lm[R_SHOULDER].x) / 2
    s_mid_y = (lm[L_SHOULDER].y + lm[R_SHOULDER].y) / 2
    h_mid_x = (lm[L_HIP].x + lm[R_HIP].x) / 2
    h_mid_y = (lm[L_HIP].y + lm[R_HIP].y) / 2
    torso_px = math.hypot((s_mid_x - h_mid_x) * img_w, (s_mid_y - h_mid_y) * img_h)
    torso_cm = round(torso_px * scale, 1)

    arm_px = _pixel_dist(lm[L_SHOULDER], lm[L_ELBOW], img_w, img_h) + _pixel_dist(lm[L_ELBOW], lm[L_WRIST], img_w, img_h)
    arm_cm = round(arm_px * scale, 1)

    leg_px = _pixel_dist(lm[L_HIP], lm[L_KNEE], img_w, img_h) + _pixel_dist(lm[L_KNEE], lm[L_ANKLE], img_w, img_h)
    leg_cm = round(leg_px * scale, 1)

    # Anthropometric 3D circumference modeling
    is_female = str(gender).lower() in ["female", "woman", "f"]
    if is_female:
        chest_w = shoulder_cm * 0.78
        chest_cm = round(_circumference_from_width(chest_w, ratio=0.72), 1)
        waist_w = max(hip_w_cm * 0.88, chest_w * 0.78)
        waist_cm = round(_circumference_from_width(waist_w, ratio=0.70), 1)
        hip_cm = round(_circumference_from_width(hip_w_cm * 1.22, ratio=0.76), 1)
    else:
        chest_w = shoulder_cm * 0.75
        chest_cm = round(_circumference_from_width(chest_w, ratio=0.68), 1)
        waist_w = max(hip_w_cm * 0.92, chest_w * 0.82)
        waist_cm = round(_circumference_from_width(waist_w, ratio=0.72), 1)
        hip_cm = round(_circumference_from_width(hip_w_cm * 1.18, ratio=0.74), 1)

    return {
        "shoulder_cm": shoulder_cm,
        "chest_cm": chest_cm,
        "waist_cm": waist_cm,
        "hip_cm": hip_cm,
        "hip_width_cm": hip_w_cm,
        "torso_cm": torso_cm,
        "arm_cm": arm_cm,
        "leg_cm": leg_cm,
        "height_cm": user_height_cm,
        "scale_factor": round(scale, 4),
        "landmarks_detected": True,
        "pose_visibility": round(visibility, 3),
        "_lm": lm,   # pose landmarks for the face crop; removed before the result is returned
        "full_body_visible": bool(is_full_body),
        "image_width": img_w,
        "image_height": img_h,
    }


def run_opencv_contour_extraction(
    image_bytes: bytes,
    user_height_cm: float = 175.0,
    gender: str = "unisex"
) -> Optional[Dict[str, Any]]:
    if not _HAS_CV2:
        return None
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        c = max(contours, key=cv2.contourArea)
        bx, by, bw, bh = cv2.boundingRect(c)
        if bh < h * 0.3 or bw < w * 0.1:
            return None
        scale = user_height_cm / float(bh)
        shoulder_cm = round(bw * scale * 0.85, 1)
        chest_cm = round(_circumference_from_width(shoulder_cm * 0.75, 0.70), 1)
        waist_cm = round(chest_cm * 0.82, 1)
        hip_cm = round(chest_cm * 0.95, 1)
        return {
            "shoulder_cm": shoulder_cm,
            "chest_cm": chest_cm,
            "waist_cm": waist_cm,
            "hip_cm": hip_cm,
            "height_cm": user_height_cm,
            "scale_factor": round(scale, 4),
            "landmarks_detected": False,  # silhouette box only, no body landmarks
            "method": "opencv_contour",
            "image_width": w,
            "image_height": h,
        }
    except Exception:
        return None


def run_fallback_extraction(
    image_bytes: bytes,
    user_height_cm: float = 175.0,
    gender: str = "unisex"
) -> Dict[str, Any]:
    w, h = 384, 512
    try:
        img = Image.open(io.BytesIO(image_bytes))
        w, h = img.size
    except Exception:
        pass

    is_female = str(gender).lower() in ["female", "woman", "f"]
    if is_female:
        base = {"shoulder_cm": 38.5, "chest_cm": 89.0, "waist_cm": 72.0, "hip_cm": 95.0}
    else:
        base = {"shoulder_cm": 44.5, "chest_cm": 96.0, "waist_cm": 79.0, "hip_cm": 94.0}

    # Scale base slightly with user height
    height_ratio = user_height_cm / 175.0
    scaled_base = {k: round(v * height_ratio, 1) for k, v in base.items()}

    return {
        **scaled_base,
        "height_cm": user_height_cm,
        "landmarks_detected": False,
        "image_width": w,
        "image_height": h,
    }


def analyze_person_image(
    image_bytes: bytes,
    user_height_cm: float = 175.0,
    gender: str = "unisex",
    age: Optional[float] = None,
    height_source: str = "user",
    face_box: Optional[Tuple[int, int, int, int]] = None,
) -> Dict[str, Any]:
    """
    Main entry point for computer vision body measurement and fashion analysis.
    Returns rich dictionary with body_type, measurements, size, and recommendations.
    `age` (e.g. from the age service) switches children to kidswear sizing.
    """
    if not image_bytes or len(image_bytes) == 0:
        raise ValueError("Image bytes cannot be empty")

    method = "simulated"
    measurements = None

    if _HAS_MEDIAPIPE and len(image_bytes) > 500:
        try:
            measurements = run_mediapipe_extraction(image_bytes, user_height_cm, gender)
            method = "mediapipe_real"
        except Exception:
            measurements = run_opencv_contour_extraction(image_bytes, user_height_cm, gender)
            if measurements:
                method = "opencv_contour"

    if measurements is None:
        measurements = run_fallback_extraction(image_bytes, user_height_cm, gender)

    shoulder = measurements.get("shoulder_cm", 44.5)
    chest = measurements.get("chest_cm", 96.0)
    waist = measurements.get("waist_cm", 79.0)
    hip = measurements.get("hip_cm", 94.0)

    if is_child(age):
        body_type = "Child build"
        size = get_kids_size(user_height_cm)
    else:
        body_type = classify_body_type(shoulder, waist, hip)
        size = get_clothing_size(chest, gender=gender)
    clothes, shoes, glasses, face = get_recommendations(body_type, size)
    pose_lm = measurements.pop("_lm", None)

    # Face shape + skin tone from face landmarks (face_analyzer.py); falls back to a fixed patch
    tone_rgb = (205.0, 160.0, 135.0)
    face_info: Dict[str, Any] = {"face_detected": False, "face_shape": "Not detected"}
    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")  # same orientation as the services
        w, h = img.size
        rgb = np.array(img)
        if max(w, h) > 1600 and _HAS_CV2:     # keep landmarking fast on large photos
            k = 1600 / max(w, h)
            rgb = cv2.resize(rgb, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
            face_box_scaled = [int(v * k) for v in face_box] if face_box else None
        else:
            face_box_scaled = face_box
        face_info = analyze_face(rgb, face_box_scaled, pose_lm)
        if face_info.get("face_detected") and face_info.get("cheek_rgb"):
            tone_rgb = face_info["cheek_rgb"]
        else:
            if face_box:  # cheeks: middle of the lower half of the detected face
                x0, y0, x1, y1 = face_box
                cx, cy = (x0 + x1) // 2, y0 + int((y1 - y0) * 0.62)
                r = max(4, (x1 - x0) // 8)
            else:
                cx, cy, r = w // 2, h // 4, 15
            patch = img.crop((max(0, cx - r), max(0, cy - r), min(w, cx + r), min(h, cy + r)))
            arr = np.array(patch)
            if arr.size > 0:
                tone_rgb = tuple(arr.mean(axis=(0, 1)))
    except Exception:
        pass
    if face_info.get("face_detected"):
        face = face_info["face_shape"]
        glasses = [f"{style} sunglasses" for style in face_info.get("frame_advice", [])[:3]] or glasses
    elif not is_child(age):
        face = "Not detected"

    skin_tone, undertone = _classify_skin_tone(tone_rgb)
    recommended_colors = _recommend_palette(undertone)

    confidence = measurement_confidence(method, measurements, height_source)

    img_w = measurements.get("image_width", 384)
    img_h = measurements.get("image_height", 512)

    return {
        "success": True,
        "analysis_method": method,
        "image_width": img_w,
        "image_height": img_h,
        "body_type": body_type,
        "face_shape": face,
        "face_analysis": {k: v for k, v in face_info.items() if k != "cheek_rgb"},
        "skin_tone": skin_tone,
        "undertone": undertone,
        "clothing_size": size,
        "size_system": "kids" if is_child(age) else "adult",
        "fit_confidence": confidence["score"],
        "measurement_confidence": confidence,
        "landmarks_detected": measurements.get("landmarks_detected", False),
        "body_measurements": {
            **measurements,
            "body_type": body_type,
            "clothing_size": size,
            "user_height_cm": user_height_cm,
            "gender": gender,
        },
        "recommended_clothes": clothes,
        "recommended_shoes": shoes,
        "recommended_sunglasses": glasses,
        "recommended_colors": recommended_colors,
    }


def analyze_image_bytes(
    image_bytes: bytes,
    user_height_cm: float = 175.0,
    gender: str = "unisex",
    age: Optional[float] = None,
    height_source: str = "user",
    face_box: Optional[Tuple[int, int, int, int]] = None,
) -> Dict[str, Any]:
    """Compatibility wrapper for virtual_tryon_3d test suite."""
    if not image_bytes or len(image_bytes) == 0:
        raise ValueError("Image bytes cannot be empty")

    return analyze_person_image(image_bytes, user_height_cm=user_height_cm, gender=gender,
                                age=age, height_source=height_source, face_box=face_box)
