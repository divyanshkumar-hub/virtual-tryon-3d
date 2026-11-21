"""Compare VisionFit-AI's MobileNetV2 clothing classifier with FashionCLIP on the same
labelled images (services/eval_data/fashion + a second sample), using VisionFit's 10 classes.

  KERAS_BACKEND=torch services/.venv/Scripts/python services/eval/compare_visionfit.py <path-to-VisionFit-AI>
"""
import io
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")
import keras  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import requests  # noqa: E402
from PIL import Image  # noqa: E402

REPO = Path(sys.argv[1])
ROOT = Path(__file__).resolve().parents[1]
MAP = {"Tshirts": "t-shirt", "Shirts": "shirt", "Jeans": "pants", "Trousers": "pants", "Track Pants": "pants",
       "Shorts": "shorts", "Skirts": "skirt", "Dresses": "dress", "Jackets": "outwear", "Blazers": "outwear",
       "Sweatshirts": "longsleeve", "Sweaters": "longsleeve", "Casual Shoes": "shoes", "Sports Shoes": "shoes",
       "Formal Shoes": "shoes", "Sandals": "shoes", "Heels": "shoes", "Flip Flops": "shoes", "Caps": "hat"}
CLASSES = json.loads((REPO / "recommendation_engine/models/class_names.json").read_text())

# balanced sample: up to 40 per VisionFit class from the parquet shard
random.seed(3)
rows = [r for r in pq.read_table(ROOT / "eval_data/raw/fashion0.parquet").to_pylist() if r["articleType"] in MAP]
by = {}
for r in rows:
    by.setdefault(MAP[r["articleType"]], []).append(r)
sample = [r for c in CLASSES for r in random.sample(by.get(c, []), min(40, len(by.get(c, []))))]
print("eval images:", len(sample), Counter(MAP[r["articleType"]] for r in sample))

model = keras.saving.load_model(REPO / "recommendation_engine/models/clothing_classifier.keras", safe_mode=True)
LABEL_TEXT = {"dress": "dress", "hat": "hat or cap", "longsleeve": "long sleeve sweater or sweatshirt", "outwear": "jacket or coat",
              "pants": "pants or jeans", "shirt": "button-up shirt", "shoes": "shoes", "shorts": "shorts", "skirt": "skirt",
              "t-shirt": "t-shirt"}

hits = {"visionfit": 0, "fashionclip": 0}
per = {c: Counter() for c in CLASSES}
conf = Counter()
for r in sample:
    truth = MAP[r["articleType"]]
    img = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
    x = np.asarray(img.resize((224, 224)), dtype="float32")[None]     # predict.py feeds raw 0-255 pixels
    vf = CLASSES[int(np.argmax(np.asarray(model.predict(x, verbose=0))[0]))]
    buf = io.BytesIO(); img.save(buf, "JPEG")
    emb = requests.post("http://127.0.0.1:5103/embed", files={"file": ("g.jpg", buf.getvalue(), "image/jpeg")}).json()["embedding"]
    per[truth]["n"] += 1
    for name, pred in (("visionfit", vf),):
        hits[name] += pred == truth; per[truth][name] += pred == truth
        if pred != truth: conf[f"{truth}->{pred}"] += 1
    r["_emb"] = np.asarray(emb)

# FashionCLIP zero-shot on the same 10 labels (text embeddings via /search is item-based, so embed labels here)
from transformers import CLIPModel, CLIPProcessor  # noqa: E402
import torch  # noqa: E402
m = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip").eval(); p = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
with torch.no_grad():
    t = m.get_text_features(**p(text=[f"a photo of a {LABEL_TEXT[c]}" for c in CLASSES], return_tensors="pt", padding=True))
    t = torch.nn.functional.normalize(t, dim=-1).numpy()
for r in sample:
    truth = MAP[r["articleType"]]
    pred = CLASSES[int(np.argmax(t @ r["_emb"]))]
    hits["fashionclip"] += pred == truth; per[truth]["fashionclip"] += pred == truth

n = len(sample)
print(f"\nTop-1 accuracy on {n} images, VisionFit's 10 classes:")
for k, v in hits.items():
    print(f"  {k:12s} {v / n:.1%}")
print("\nper class (VisionFit / FashionCLIP):")
for c in CLASSES:
    if per[c]["n"]:
        print(f"  {c:10s} n={per[c]['n']:3d}  {per[c]['visionfit'] / per[c]['n']:.0%} / {per[c]['fashionclip'] / per[c]['n']:.0%}")
print("\nVisionFit top confusions:", conf.most_common(6))
