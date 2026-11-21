"""Port 5103 - FashionCLIP: garment tagging (type / colour / fabric / style),
embeddings and catalog recommendations.

Run:  services/.venv/Scripts/python -m uvicorn fashion_service:app --port 5103 --app-dir services
"""
from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import torch
from fastapi import Body, File, Form, UploadFile
from transformers import CLIPModel, CLIPProcessor

from common import DEVICE, make_app, read_image, timer

MODEL_ID = os.environ.get("FASHION_CLIP_MODEL", "patrickjohncyh/fashion-clip")

GARMENT_TYPES = [
    "t-shirt", "shirt", "polo shirt", "sweater", "hoodie", "jacket", "blazer", "coat",
    "dress", "skirt", "jeans", "trousers", "shorts", "track pants", "kurta", "saree",
    "sneakers", "formal shoes", "sandals", "heels", "handbag", "watch", "sunglasses", "cap",
]
COLOURS = [
    "black", "white", "grey", "navy blue", "blue", "red", "maroon", "pink", "purple",
    "green", "olive", "yellow", "orange", "brown", "beige", "cream", "gold", "silver", "multicolour",
]
FABRICS = [
    "cotton", "denim", "linen", "silk", "satin", "wool", "knit", "leather", "suede",
    "polyester", "chiffon", "velvet", "fleece", "corduroy", "lace", "nylon",
]
# simulation hints for the 3D cloth layer (rough XPBD-style presets)
FABRIC_SIM = {
    "cotton": {"stiffness": 0.55, "bend": 0.35, "mass_gsm": 150, "friction": 0.5},
    "denim": {"stiffness": 0.85, "bend": 0.70, "mass_gsm": 350, "friction": 0.6},
    "linen": {"stiffness": 0.65, "bend": 0.45, "mass_gsm": 180, "friction": 0.55},
    "silk": {"stiffness": 0.20, "bend": 0.08, "mass_gsm": 60, "friction": 0.2},
    "satin": {"stiffness": 0.25, "bend": 0.10, "mass_gsm": 90, "friction": 0.2},
    "wool": {"stiffness": 0.60, "bend": 0.50, "mass_gsm": 280, "friction": 0.65},
    "knit": {"stiffness": 0.30, "bend": 0.25, "mass_gsm": 200, "friction": 0.6},
    "leather": {"stiffness": 0.95, "bend": 0.85, "mass_gsm": 900, "friction": 0.7},
    "suede": {"stiffness": 0.85, "bend": 0.75, "mass_gsm": 700, "friction": 0.75},
    "polyester": {"stiffness": 0.45, "bend": 0.25, "mass_gsm": 120, "friction": 0.35},
    "chiffon": {"stiffness": 0.10, "bend": 0.04, "mass_gsm": 40, "friction": 0.25},
    "velvet": {"stiffness": 0.50, "bend": 0.40, "mass_gsm": 300, "friction": 0.8},
    "fleece": {"stiffness": 0.40, "bend": 0.35, "mass_gsm": 260, "friction": 0.75},
    "corduroy": {"stiffness": 0.70, "bend": 0.55, "mass_gsm": 320, "friction": 0.7},
    "lace": {"stiffness": 0.15, "bend": 0.06, "mass_gsm": 70, "friction": 0.3},
    "nylon": {"stiffness": 0.35, "bend": 0.15, "mass_gsm": 80, "friction": 0.25},
}
STYLES = ["casual", "formal", "sportswear", "party wear", "ethnic wear", "streetwear", "business casual"]

app = make_app("FashionCLIP service")

_model = CLIPModel.from_pretrained(MODEL_ID).to(DEVICE).eval()
_proc = CLIPProcessor.from_pretrained(MODEL_ID)


@torch.no_grad()
def embed_images(images) -> np.ndarray:
    inputs = _proc(images=images, return_tensors="pt").to(DEVICE)
    feats = _model.get_image_features(**inputs)
    feats = feats if isinstance(feats, torch.Tensor) else feats.pooler_output
    return torch.nn.functional.normalize(feats, dim=-1).cpu().numpy()


@torch.no_grad()
def embed_texts(texts: List[str]) -> np.ndarray:
    inputs = _proc(text=texts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    feats = _model.get_text_features(**inputs)
    feats = feats if isinstance(feats, torch.Tensor) else feats.pooler_output
    return torch.nn.functional.normalize(feats, dim=-1).cpu().numpy()


_label_cache: Dict[str, np.ndarray] = {}


def _labels(name: str, labels: List[str], template: str) -> np.ndarray:
    if name not in _label_cache:
        _label_cache[name] = embed_texts([template.format(l) for l in labels])
    return _label_cache[name]


def zero_shot(img_emb: np.ndarray, name: str, labels: List[str], template: str, k: int = 3):
    logits = 100.0 * img_emb @ _labels(name, labels, template).T
    probs = np.exp(logits - logits.max(-1, keepdims=True))
    probs /= probs.sum(-1, keepdims=True)
    order = np.argsort(-probs[0])[:k]
    return [{"label": labels[i], "p": round(float(probs[0, i]), 3)} for i in order]


def tag_embedding(emb: np.ndarray) -> dict:
    fabric = zero_shot(emb, "fabric", FABRICS, "a garment made of {} fabric")
    return {
        "type": zero_shot(emb, "type", GARMENT_TYPES, "a photo of a {}"),
        "colour": zero_shot(emb, "colour", COLOURS, "a {} coloured garment"),
        "fabric": fabric,
        "fabric_sim": FABRIC_SIM[fabric[0]["label"]],
        "style": zero_shot(emb, "style", STYLES, "a {} outfit"),
    }


# ---------------------------------------------------------------- catalog
# In-memory catalog: id -> {embedding, meta}. The main app would persist the
# vectors in SQLite; for the experiment memory is enough.
CATALOG: Dict[str, dict] = {}


@app.get("/health")
def health():
    return {"ok": True, "model": MODEL_ID, "device": DEVICE, "catalog_items": len(CATALOG)}


@app.post("/catalog/reset")
def catalog_reset():
    CATALOG.clear()
    return {"ok": True}


@app.post("/tag")
async def tag(file: UploadFile = File(...)):
    img = await read_image(file)
    out: dict = {}
    with timer(out):
        out.update(tag_embedding(embed_images([img])))
    return out


@app.post("/embed")
async def embed(file: UploadFile = File(...)):
    img = await read_image(file)
    return {"embedding": embed_images([img])[0].round(5).tolist()}


@app.post("/catalog/add")
async def catalog_add(item_id: str = Form(...), name: str = Form(""), category: str = Form(""),
                      file: UploadFile = File(...)):
    img = await read_image(file)
    emb = embed_images([img])
    CATALOG[item_id] = {"emb": emb[0], "meta": {"id": item_id, "name": name, "category": category,
                                                 **{k: v[0]["label"] for k, v in tag_embedding(emb).items()
                                                    if isinstance(v, list)}}}
    return CATALOG[item_id]["meta"]


def _rank(query: np.ndarray, k: int, exclude=(), category=None, allowed_colours=None, boost=0.0):
    ids = [i for i in CATALOG if i not in exclude
           and (category is None or CATALOG[i]["meta"]["category"] == category)]
    if not ids:
        return []
    mat = np.stack([CATALOG[i]["emb"] for i in ids])
    scores = mat @ query
    if allowed_colours:  # soft re-rank with the rule-based palette from vision_analyzer
        scores = scores + boost * np.array([CATALOG[i]["meta"]["colour"] in allowed_colours for i in ids])
    order = np.argsort(-scores)[:k]
    return [{**CATALOG[ids[j]]["meta"], "score": round(float(scores[j]), 4)} for j in order]


@app.post("/search")
def search(q: str = Body(..., embed=True), k: int = Body(8, embed=True)):
    """Free-text search, e.g. 'linen shirt for a summer wedding'."""
    return _rank(embed_texts([q])[0], k)


@app.get("/similar/{item_id}")
def similar(item_id: str, k: int = 8):
    return _rank(CATALOG[item_id]["emb"], k, exclude={item_id})


@app.post("/recommend")
def recommend(liked_ids: List[str] = Body([], embed=True), occasion: str = Body("", embed=True),
              palette: List[str] = Body([], embed=True), category: str | None = Body(None, embed=True),
              k: int = Body(8, embed=True)):
    """Profile vector = mean(liked items) + occasion text; palette colours get a boost."""
    parts = [CATALOG[i]["emb"] for i in liked_ids if i in CATALOG]
    if occasion:
        parts.append(embed_texts([occasion])[0])
    if not parts:
        return []
    q = np.mean(parts, axis=0)
    q /= np.linalg.norm(q)
    pal = [p.lower() for p in palette]
    return _rank(q, k, exclude=set(liked_ids), category=category, allowed_colours=pal, boost=0.05)
