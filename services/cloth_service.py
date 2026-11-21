"""Port 5104 - cloth analysis: garment photo -> clothing type.

The model is Akshit Kandari's VisionFit-AI clothing classifier
(github.com/akshitkandari63-cloud/VisionFit-AI, recommendation_engine/): MobileNetV2 (ImageNet)
with a 10-class head trained on clothing-dataset-small. Copied unchanged into
models/visionfit_cloth/.

It was trained with TensorFlow, but the .keras file only uses standard layers, so Keras 3 runs
it here on the PyTorch backend that the other services already use. On our test images the
torch and TensorFlow predictions agree to about 0.005. Preprocessing matches VisionFit's
predict.py: nearest-neighbour resize to 224x224, raw 0-255 RGB (the model scales inputs itself).

Run:  services/.venv/Scripts/python -m uvicorn cloth_service:app --port 5104 --app-dir services
"""
from __future__ import annotations

import json
import os

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("KERAS_TORCH_DEVICE", "cpu")  # ~70 ms per image; leaves the 4 GB GPU to FashionCLIP

import keras  # noqa: E402
import numpy as np  # noqa: E402
from fastapi import File, UploadFile  # noqa: E402
from PIL import Image  # noqa: E402

from common import make_app, read_image, timer  # noqa: E402

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "visionfit_cloth")
IMAGE_SIZE = (224, 224)

model = keras.saving.load_model(os.path.join(MODEL_DIR, "clothing_classifier.keras"))
with open(os.path.join(MODEL_DIR, "class_names.json"), encoding="utf-8") as f:
    CLASS_NAMES = json.load(f)
if model.output_shape[-1] != len(CLASS_NAMES):
    raise RuntimeError("clothing_classifier.keras and class_names.json disagree on the number of classes")

app = make_app("Cloth analysis (VisionFit-AI classifier)")


@app.get("/health")
def health():
    return {"ok": True, "model": "VisionFit-AI clothing_classifier (MobileNetV2)", "backend": keras.backend.backend(),
            "device": os.environ["KERAS_TORCH_DEVICE"], "classes": CLASS_NAMES}


@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    img = await read_image(file)
    out = {}
    with timer(out):
        x = np.asarray(img.resize(IMAGE_SIZE, Image.NEAREST), dtype="float32")[None]
        probs = np.asarray(keras.ops.convert_to_numpy(model(x, training=False)))[0]
    order = np.argsort(probs)[::-1]
    out["predictions"] = [{"label": CLASS_NAMES[int(i)], "score": round(float(probs[i]), 4)} for i in order]
    out["category"] = out["predictions"][0]["label"]
    out["score"] = out["predictions"][0]["score"]
    return out
