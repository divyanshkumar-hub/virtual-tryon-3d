"""Build height test cases from local images (nothing leaves the machine).

Ground truth for the marker test: the person's pixel height in the full-body
photo was annotated by hand (hair top y=19, sole y~958 -> 939 px, +/-5 px).
We *assume* a true height of 110 cm (plausible for the child's age) and paste a
15 cm ArUco marker at the matching scale onto the wall at body depth. If mask
extent + marker detection are right, the service returns ~110 cm.
"""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
CHILD = ROOT / "uploads" / "0911228c-2e88-43f3-91a3-8819d7821c4b_d.jpeg"
RENDER = ROOT / "uploads" / "645ca06a-d372-41e9-9a51-0aafc58067f0_uploaded_person.jpg"
CARTOON = ROOT / "samples" / "1_target_person.jpg"
ANNOT_PX, ASSUMED_CM, MARKER_CM = 939, 110.0, 15.0


def _marker(side_px: int) -> Image.Image:
    m = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 7, side_px)
    q = side_px // 6  # white quiet zone, like a printed sheet
    return Image.fromarray(cv2.copyMakeBorder(m, q, q, q, q, cv2.BORDER_CONSTANT, value=255)).convert("RGB")


def make_cases(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    child = Image.open(CHILD).convert("RGB")
    side = round(MARKER_CM * ANNOT_PX / ASSUMED_CM)  # ~128 px
    with_marker = child.copy()
    with_marker.paste(_marker(side), (20, 420))
    p = out / "child_marker.jpg"
    with_marker.save(p, quality=92)
    small = with_marker.resize((360, 480), Image.LANCZOS)
    p_small = out / "child_marker_360x480_q50.jpg"
    small.save(p_small, quality=50)
    p_plain = out / "child_plain.jpg"
    child.save(p_plain, quality=95)
    p_render = out / "render_partial_body.jpg"
    Image.open(RENDER).convert("RGB").resize((1920, 1080)).save(p_render, quality=90)
    return [
        {"name": "marker_full_res", "path": p, "truth_cm": ASSUMED_CM, "truth_px": ANNOT_PX},
        {"name": "marker_low_res_jpeg50", "path": p_small, "truth_cm": ASSUMED_CM, "truth_px": ANNOT_PX / 2},
        {"name": "user_height_only", "path": p_plain, "user_height_cm": ASSUMED_CM, "truth_px": ANNOT_PX},
        {"name": "depth_only_no_marker", "path": p_plain, "note": "plausible range for a ~4-6 y/o: 100-120 cm"},
        {"name": "partial_body_game_render", "path": p_render, "note": "should flag full_body_visible=false"},
        {"name": "cartoon_sample", "path": CARTOON, "user_height_cm": 175, "note": "legs cut off; synthetic"},
    ]
