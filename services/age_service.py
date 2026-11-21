"""Port 5101 - age (and gender) estimation.

Primary : MiVOLO v2 (iitolstykh/mivolo_v2, CC-BY-NC-4.0 -> NON-COMMERCIAL ONLY).
          Uses face crop + body crop together, so it still works when the face
          is small in a full-body photo.
Baseline: nateraw/vit-age-classifier (face-only, 9 age buckets).
Face box : MediaPipe BlazeFace; body box: MediaPipe pose segmentation mask.

Run:  services/.venv/Scripts/python -m uvicorn age_service:app --port 5101 --app-dir services
"""
from __future__ import annotations

import os
from pathlib import Path

import mediapipe as mp
import numpy as np
import torch
from fastapi import File, Form, UploadFile
from PIL import Image
from transformers import (AutoConfig, AutoImageProcessor, AutoModelForImageClassification,
                          ViTForImageClassification, ViTImageProcessor)

from common import DEVICE, make_app, read_image, timer

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
app = make_app("Age service")

DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
_cfg = AutoConfig.from_pretrained("iitolstykh/mivolo_v2", trust_remote_code=True)
_mivolo = AutoModelForImageClassification.from_pretrained(
    "iitolstykh/mivolo_v2", trust_remote_code=True, torch_dtype=DTYPE).to(DEVICE).eval()
_mivolo_proc = AutoImageProcessor.from_pretrained("iitolstykh/mivolo_v2", trust_remote_code=True)

_vit = ViTForImageClassification.from_pretrained("nateraw/vit-age-classifier").to(DEVICE).eval()
_vit_proc = ViTImageProcessor.from_pretrained("nateraw/vit-age-classifier")
VIT_MID = {"0-2": 1, "3-9": 6, "10-19": 15, "20-29": 25, "30-39": 35, "40-49": 45,
           "50-59": 55, "60-69": 65, "more than 70": 75}

_face = mp.tasks.vision.FaceDetector.create_from_options(mp.tasks.vision.FaceDetectorOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=str(MODELS / "blaze_face_short_range.tflite")),
    min_detection_confidence=0.4))
_pose = mp.tasks.vision.PoseLandmarker.create_from_options(mp.tasks.vision.PoseLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=str(MODELS / "pose_landmarker_heavy.task")),
    output_segmentation_masks=True))


def _face_box(rgb: np.ndarray):
    h, w = rgb.shape[:2]
    res = _face.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not res.detections:
        return None
    b = max(res.detections, key=lambda d: d.bounding_box.width * d.bounding_box.height).bounding_box
    # BlazeFace boxes are tight; pad 25% so hair/jaw are included like MiVOLO's training crops
    pad_w, pad_h = b.width * 0.25, b.height * 0.25
    return (max(0, int(b.origin_x - pad_w)), max(0, int(b.origin_y - pad_h * 1.4)),
            min(w, int(b.origin_x + b.width + pad_w)), min(h, int(b.origin_y + b.height + pad_h)))


def _body_box(rgb: np.ndarray):
    res = _pose.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not res.segmentation_masks:
        return None
    m = res.segmentation_masks[0].numpy_view().squeeze() > 0.5
    ys, xs = np.where(m)
    if ys.size < 100:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _crop(rgb, box):
    return None if box is None else rgb[box[1]:box[3], box[0]:box[2]][:, :, ::-1].copy()  # MiVOLO wants BGR


@torch.no_grad()
def run_mivolo(rgb, fbox, bbox):
    faces, bodies = [_crop(rgb, fbox)], [_crop(rgb, bbox)]
    if faces[0] is None and bodies[0] is None:
        return None
    to = dict(dtype=_mivolo.dtype, device=_mivolo.device)
    f_in = _mivolo_proc(images=faces)["pixel_values"].to(**to)
    b_in = _mivolo_proc(images=bodies)["pixel_values"].to(**to)
    out = _mivolo(faces_input=f_in, body_input=b_in)
    return {"age": round(float(out.age_output[0].item()), 1),
            "gender": _cfg.gender_id2label[out.gender_class_idx[0].item()],
            "gender_p": round(float(out.gender_probs[0].item()), 3)}


@torch.no_grad()
def run_vit(rgb, fbox):
    img = Image.fromarray(rgb if fbox is None else rgb[fbox[1]:fbox[3], fbox[0]:fbox[2]])
    probs = _vit(**_vit_proc(img, return_tensors="pt").to(DEVICE)).logits.softmax(-1)[0].cpu().numpy()
    labels = [_vit.config.id2label[i] for i in range(len(probs))]
    return {"bucket": labels[int(probs.argmax())], "p": round(float(probs.max()), 3),
            "age": round(float(sum(p * VIT_MID[l] for p, l in zip(probs, labels))), 1)}


@app.get("/health")
def health():
    return {"ok": True, "device": DEVICE, "models": ["iitolstykh/mivolo_v2 (CC-BY-NC-4.0)", "nateraw/vit-age-classifier"]}


@app.post("/estimate")
async def estimate(file: UploadFile = File(...), mode: str = Form("auto")):
    """mode: auto (detect face+body) | face (image is already a face crop) | body"""
    rgb = np.ascontiguousarray(np.asarray(await read_image(file)))
    out: dict = {}
    with timer(out):
        h, w = rgb.shape[:2]
        if mode == "face":
            fbox, bbox = (0, 0, w, h), None
        else:
            fbox, bbox = _face_box(rgb), _body_box(rgb)
            if mode == "body":
                fbox = None
        out.update(face_box=fbox, body_box=bbox, mivolo=run_mivolo(rgb, fbox, bbox),
                   vit_baseline=run_vit(rgb, fbox))
        a = (out["mivolo"] or {}).get("age")
        out["age_range"] = None if a is None else [max(0, round(a - 4)), round(a + 4)]
    return out
