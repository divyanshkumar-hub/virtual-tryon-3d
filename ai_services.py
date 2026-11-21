"""HTTP client for the optional AI model services in services/ (ports 5101-5104).

Every call is best-effort: on timeout / connection error / bad response it
returns None, and mini.py falls back to the local vision_analyzer. Set
AI_SERVICES_DISABLED=1 to force the fallback path (used by the test suite).
"""
from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, List, Optional

import requests

URLS = {
    "age": os.environ.get("AGE_SERVICE_URL", "http://127.0.0.1:5101"),
    "height": os.environ.get("HEIGHT_SERVICE_URL", "http://127.0.0.1:5102"),
    "fashion": os.environ.get("FASHION_SERVICE_URL", "http://127.0.0.1:5103"),
    "cloth": os.environ.get("CLOTH_SERVICE_URL", "http://127.0.0.1:5104"),
}
DESCRIPTIONS = {
    "age": "Age & gender (MiVOLO v2)",
    "height": "Height & measurements (pose segmentation + ArUco marker)",
    "fashion": "Garment tagging & recommendations (FashionCLIP)",
    "cloth": "Cloth analysis (VisionFit-AI clothing classifier, Akshit Kandari)",
}

_status_cache: Dict[str, Any] = {"t": 0.0, "data": {}}
_lock = threading.Lock()


def disabled() -> bool:
    return os.environ.get("AI_SERVICES_DISABLED") == "1"


def _post(svc: str, path: str, timeout: float, files=None, data=None, json=None) -> Optional[Any]:
    if disabled():
        return None
    try:
        r = requests.post(URLS[svc] + path, files=files, data=data, json=json, timeout=timeout)
        if r.status_code == 200:
            return r.json()
        return {"_error": r.status_code, "detail": _detail(r)}
    except requests.RequestException:
        return None


def _get(svc: str, path: str, timeout: float) -> Optional[Any]:
    if disabled():
        return None
    try:
        r = requests.get(URLS[svc] + path, timeout=timeout)
        return r.json() if r.status_code == 200 else None
    except requests.RequestException:
        return None


def _detail(r) -> str:
    try:
        return str(r.json().get("detail", ""))
    except Exception:  # noqa: BLE001
        return r.text[:200]


def ok(res: Any) -> bool:
    return isinstance(res, (dict, list)) and not (isinstance(res, dict) and "_error" in res)


# ------------------------------------------------------------------ status
def status(max_age: float = 5.0) -> Dict[str, Any]:
    """Health of every service, cached for a few seconds."""
    with _lock:
        if time.time() - _status_cache["t"] < max_age:
            return _status_cache["data"]
    out = {}
    for name in URLS:
        h = _get(name, "/health", timeout=0.8)
        out[name] = {"online": h is not None, "url": URLS[name], "description": DESCRIPTIONS[name], "health": h}
    with _lock:
        _status_cache.update(t=time.time(), data=out)
    return out


def online(name: str, fresh: bool = False) -> bool:
    return status(max_age=0 if fresh else 5.0).get(name, {}).get("online", False)


# ------------------------------------------------------------------ person analysis
def estimate_age(image: bytes) -> Optional[Dict[str, Any]]:
    res = _post("age", "/estimate", 20, files={"file": ("p.jpg", image, "image/jpeg")}, data={"mode": "auto"})
    return res if ok(res) else None


def estimate_height(image: bytes, user_height_cm: Optional[float] = None,
                    marker_cm: Optional[float] = None) -> Optional[Dict[str, Any]]:
    data = {"use_depth": "false"}  # monocular depth was ~2.5x off in our tests - never used for sizing
    if user_height_cm:
        data["user_height_cm"] = str(user_height_cm)
    if marker_cm:
        data["marker_cm"] = str(marker_cm)
    res = _post("height", "/estimate", 30, files={"file": ("p.jpg", image, "image/jpeg")}, data=data)
    return res if ok(res) else (res if isinstance(res, dict) and res.get("_error") == 422 else None)


# ------------------------------------------------------------------ catalog / recommendations
def tag_garment(image: bytes) -> Optional[Dict[str, Any]]:
    res = _post("fashion", "/tag", 20, files={"file": ("g.jpg", image, "image/jpeg")})
    return res if ok(res) else None


def classify_cloth(image: bytes) -> Optional[Dict[str, Any]]:
    """VisionFit-AI classifier: {category, score, predictions:[{label, score}]} over its 10 clothing types."""
    res = _post("cloth", "/classify", 15, files={"file": ("g.jpg", image, "image/jpeg")})
    return res if ok(res) else None


def catalog_add(item_id: str, name: str, category: str, image: bytes) -> Optional[Dict[str, Any]]:
    res = _post("fashion", "/catalog/add", 20, files={"file": ("g.jpg", image, "image/jpeg")},
                data={"item_id": item_id, "name": name, "category": category})
    return res if ok(res) else None


def catalog_reset() -> bool:
    return ok(_post("fashion", "/catalog/reset", 10))


def catalog_size() -> int:
    h = status(max_age=0).get("fashion", {}).get("health") or {}
    return int(h.get("catalog_items", 0))


def search(q: str, k: int = 12) -> Optional[List[Dict[str, Any]]]:
    res = _post("fashion", "/search", 15, json={"q": q, "k": k})
    return res if isinstance(res, list) else None


def similar(item_id: str, k: int = 8) -> Optional[List[Dict[str, Any]]]:
    res = _get("fashion", f"/similar/{item_id}?k={k}", 15)
    return res if isinstance(res, list) else None


def recommend(liked_ids: List[str], occasion: str = "", palette: Optional[List[str]] = None,
              category: Optional[str] = None, k: int = 40) -> Optional[List[Dict[str, Any]]]:
    res = _post("fashion", "/recommend", 15, json={"liked_ids": liked_ids, "occasion": occasion,
                                                   "palette": palette or [], "category": category, "k": k})
    return res if isinstance(res, list) else None
