"""StyleMate backend: Flask + SQLite + AI analysis, catalog recommendations and try-on.

The heavy models run as separate services (services/, ports 5101-5103) and are
called through ai_services.py. Every feature degrades gracefully when a
service is offline: analysis falls back to vision_analyzer.py (MediaPipe /
OpenCV / averages) and recommendations fall back to rule-based scoring.
Try-on happens in the browser on the 3D avatar (static/js/avatar_studio.js).
"""
import io
import json
import os
import threading
import time
import webbrowser
from typing import Any, Dict, List, Optional
from uuid import uuid4

from flask import Flask, jsonify, redirect, request, send_file, send_from_directory
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import inspect, text
from werkzeug.utils import secure_filename

import ai_services as ai
from fit import fit_report
from vision_analyzer import analyze_image_bytes, is_child, measurement_confidence

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INSTANCE_DIR = os.path.join(BASE_DIR, "instance")
UPLOAD_FOLDER = os.environ.get("STYLEMATE_UPLOADS", os.path.join(BASE_DIR, "uploads"))
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")
STATIC_DIR = os.path.join(BASE_DIR, "static")
CATALOG_JSON = os.path.join(STATIC_DIR, "catalog", "catalog.json")
USER_GARMENTS_DIR = os.path.join(STATIC_DIR, "catalog", "user")
WEB_APP_DIR = os.path.join(BASE_DIR, "web_app")
COMBINED_HTML = os.path.join(BASE_DIR, "combined_app.html")

for d in (INSTANCE_DIR, UPLOAD_FOLDER, SAMPLES_DIR, USER_GARMENTS_DIR):
    os.makedirs(d, exist_ok=True)

app = Flask(__name__, instance_path=INSTANCE_DIR, static_folder=STATIC_DIR, static_url_path="/static")
CORS(app)

DB_PATH = os.environ.get("STYLEMATE_DB", os.path.join(INSTANCE_DIR, "stylemate.db"))
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{DB_PATH}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

db = SQLAlchemy(app)

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp"}


# =========================================================
# DATABASE TABLES
# =========================================================
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)


class Analysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    image_path = db.Column(db.String(255))
    body_measurements = db.Column(db.Text)
    face_shape = db.Column(db.String(50))
    # added in v2 (migrated in place for existing databases)
    age = db.Column(db.Float)
    gender = db.Column(db.String(20))
    height_cm = db.Column(db.Float)
    height_source = db.Column(db.String(20))
    result_json = db.Column(db.Text)


class Clothing(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    category = db.Column(db.String(50), nullable=False)
    color = db.Column(db.String(50))
    image_path = db.Column(db.String(255))
    model_path = db.Column(db.String(255))
    # added in v2
    sku = db.Column(db.String(40))
    article_type = db.Column(db.String(50))
    audience = db.Column(db.String(20))
    colour_name = db.Column(db.String(40))
    usage = db.Column(db.String(40))
    fabric = db.Column(db.String(40))
    # retail fields (v3)
    brand = db.Column(db.String(100))
    price = db.Column(db.Float)
    price_is_demo = db.Column(db.Boolean, default=False)
    currency = db.Column(db.String(8))
    available_sizes = db.Column(db.Text)      # JSON list
    size_chart = db.Column(db.Text)           # JSON {size: {chest|waist|hip: cm}}
    fit_type = db.Column(db.String(20))       # slim / regular / relaxed / ...
    pattern = db.Column(db.String(30))


class SavedOutfit(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    clothing_id = db.Column(db.Integer, db.ForeignKey("clothing.id"), nullable=False)
    accessory_id = db.Column(db.Integer)


def migrate_schema(engine=None):
    """create_all() does not add columns to existing tables - add the v2 ones."""
    engine = engine or db.engine
    insp = inspect(engine)
    for model in (Analysis, Clothing):
        table = model.__table__
        have = {c["name"] for c in insp.get_columns(table.name)}
        for col in table.columns:
            if col.name not in have:
                ddl = col.type.compile(engine.dialect)
                with engine.begin() as conn:
                    conn.execute(text(f'ALTER TABLE "{table.name}" ADD COLUMN "{col.name}" {ddl}'))


# =========================================================
# CATALOG
# =========================================================
FABRIC_BY_ARTICLE = {
    "Jeans": "denim", "Tshirts": "cotton", "Shirts": "cotton", "Tops": "cotton", "Kurtas": "cotton",
    "Sweaters": "knit", "Sweatshirts": "fleece", "Jackets": "polyester", "Blazers": "wool", "Trousers": "cotton",
    "Shorts": "cotton", "Track Pants": "polyester", "Skirts": "polyester", "Capris": "cotton", "Leggings": "knit",
    "Dresses": "polyester", "Casual Shoes": "leather", "Sports Shoes": "nylon", "Formal Shoes": "leather",
    "Sandals": "leather", "Heels": "leather", "Flip Flops": "nylon", "Handbags": "leather", "Belts": "leather",
    "Caps": "cotton", "Backpacks": "nylon",
}
FABRIC_WORDS = ["denim", "linen", "silk", "satin", "wool", "leather", "suede", "chiffon", "velvet", "fleece",
                "corduroy", "lace", "cotton", "knit", "polyester", "nylon"]
# Physically-based material hints for the three.js avatar (MeshStandardMaterial)
FABRIC_MATERIAL = {
    "cotton": {"roughness": 0.9, "metalness": 0.0}, "denim": {"roughness": 0.95, "metalness": 0.0},
    "linen": {"roughness": 0.92, "metalness": 0.0}, "silk": {"roughness": 0.35, "metalness": 0.08},
    "satin": {"roughness": 0.3, "metalness": 0.1}, "wool": {"roughness": 1.0, "metalness": 0.0},
    "knit": {"roughness": 1.0, "metalness": 0.0}, "leather": {"roughness": 0.45, "metalness": 0.05},
    "suede": {"roughness": 1.0, "metalness": 0.0}, "polyester": {"roughness": 0.6, "metalness": 0.02},
    "chiffon": {"roughness": 0.55, "metalness": 0.0}, "velvet": {"roughness": 0.85, "metalness": 0.05},
    "fleece": {"roughness": 1.0, "metalness": 0.0}, "corduroy": {"roughness": 0.95, "metalness": 0.0},
    "lace": {"roughness": 0.8, "metalness": 0.0}, "nylon": {"roughness": 0.5, "metalness": 0.03},
}
# undertone palette -> catalog colour names
PALETTE_COLOURS = {
    "Warm": ["Olive", "Brown", "Mustard", "Cream", "Rust", "Orange", "Beige", "Khaki", "Gold", "Maroon", "Tan",
             "Peach", "Coffee Brown", "Off White", "Copper", "Bronze"],
    "Cool": ["Navy Blue", "Blue", "Grey", "Charcoal", "White", "Burgundy", "Purple", "Silver", "Black", "Teal",
             "Lavender", "Magenta", "Pink", "Steel"],
    "Neutral": ["Teal", "Grey", "Green", "Pink", "Navy Blue", "White", "Black", "Beige", "Grey Melange", "Olive",
                "Rose", "Off White", "Blue"],
}
# article types that flatter a body shape (soft boost)
BODY_TYPE_BOOST = {
    "inverted": {"Jeans", "Trousers", "Track Pants", "Shorts", "Tshirts", "Kurtas", "Casual Shoes"},
    "pear": {"Blazers", "Shirts", "Jackets", "Tops", "Sweaters", "Heels", "Formal Shoes"},
    "hourglass": {"Dresses", "Blazers", "Shirts", "Trousers", "Skirts", "Heels"},
    "rectangle": {"Jackets", "Sweatshirts", "Shirts", "Jeans", "Sweaters", "Casual Shoes"},
    "oval": {"Shirts", "Blazers", "Trousers", "Kurtas", "Formal Shoes"},
}
OCCASIONS = {
    "everyday": "a comfortable casual everyday outfit",
    "office": "smart formal office wear",
    "party": "a stylish party outfit",
    "sport": "sporty athletic wear",
    "ethnic": "traditional ethnic wear",
    "date": "a smart casual date night outfit",
}
CATEGORY_GROUPS = ["tops", "outerwear", "dresses", "bottoms", "footwear", "accessories"]


def guess_fabric(name: str, article_type: Optional[str]) -> Optional[str]:
    low = (name or "").lower()
    for w in FABRIC_WORDS:
        if w in low:
            return w
    return FABRIC_BY_ARTICLE.get(article_type or "")


def _json(text: Optional[str], default):
    try:
        v = json.loads(text) if text else default
        return v if isinstance(v, type(default)) else default
    except (TypeError, ValueError):
        return default


def clothing_dict(c: "Clothing", score: Optional[float] = None, reasons: Optional[List[str]] = None,
                  with_chart: bool = False) -> Dict:
    chart = _json(c.size_chart, {})
    d = {
        "id": c.id, "name": c.name, "category": c.category, "color": c.color, "image_path": c.image_path,
        "model_path": c.model_path, "sku": c.sku, "article_type": c.article_type, "audience": c.audience,
        "colour_name": c.colour_name, "usage": c.usage, "fabric": c.fabric,
        "material": FABRIC_MATERIAL.get(c.fabric or "", {"roughness": 0.8, "metalness": 0.0}),
        "brand": c.brand, "price": c.price, "price_is_demo": bool(c.price_is_demo), "currency": c.currency or "INR",
        "available_sizes": _json(c.available_sizes, []), "fit_type": c.fit_type, "pattern": c.pattern,
        "has_size_chart": bool(chart),
    }
    if with_chart:
        d["size_chart"] = chart
    if score is not None:
        d["score"] = round(score, 4)
    if reasons is not None:
        d["reasons"] = reasons
    return d


def refresh_catalog():
    """Upsert static/catalog/catalog.json by SKU; drop old catalog rows no saved outfit uses."""
    with open(CATALOG_JSON, encoding="utf-8") as f:
        items = {it["sku"]: it for it in json.load(f)}
    existing = {c.sku: c for c in Clothing.query.filter(Clothing.sku.isnot(None)).all()}
    for sku, it in items.items():
        c = existing.get(sku) or Clothing(sku=sku, name=it["name"], category=it["category"])
        c.name, c.category, c.color, c.image_path = it["name"], it["category"], it["color"], it["image_path"]
        c.model_path = f"3d://mannequin/{it['category']}"
        c.article_type, c.audience, c.colour_name, c.usage = it["article_type"], it["audience"], it["colour_name"], it["usage"]
        c.fabric = c.fabric or guess_fabric(it["name"], it["article_type"])
        c.brand, c.price, c.currency = it.get("brand"), it.get("price"), it.get("currency", "INR")
        c.price_is_demo = bool(it.get("price_is_demo"))
        c.available_sizes = json.dumps(it.get("available_sizes", []))
        c.size_chart = json.dumps(it.get("size_chart", {}))
        c.fit_type, c.pattern = it.get("fit_type"), it.get("pattern")
        db.session.add(c)
    used = {o.clothing_id for o in SavedOutfit.query.all()}
    for sku, c in existing.items():
        if sku not in items and not sku.startswith("user-") and c.id not in used:
            db.session.delete(c)
    db.session.commit()


def seed_default_data():
    """Default user + demo catalog (static/catalog/catalog.json) in stylemate.db."""
    if not User.query.first():
        db.session.add(User(name="StyleMate Demo User", email="demo@stylemate.ai"))
        db.session.commit()

    if os.path.exists(CATALOG_JSON):
        refresh_catalog()

    if not Clothing.query.first():  # catalog.json missing: keep the original 6 demo items
        for name, cat, col in [("Navy Tailored Double-Breasted Blazer", "outerwear", "#1e3a8a"),
                               ("Cyber Urban Hoodie", "tops", "#e11d48"),
                               ("Slim-Fit White Oxford Shirt", "tops", "#f8fafc"),
                               ("Vintage Leather Rider Jacket", "outerwear", "#0f172a"),
                               ("Relaxed Straight-Leg Denim Jeans", "bottoms", "#1d4ed8"),
                               ("Tapered Chino Trousers", "bottoms", "#b45309")]:
            db.session.add(Clothing(name=name, category=cat, color=col, image_path="/samples/2_garment_cloth.jpg",
                                    model_path="3d://mannequin/custom", audience="unisex",
                                    fabric=guess_fabric(name, None)))
        db.session.commit()


# ---------------------------------------------------------------- sync with the FashionCLIP service
_sync_lock = threading.Lock()
_sync_state = {"running": False, "done": 0, "total": 0, "last": None}


def _image_bytes_for(path: Optional[str]) -> Optional[bytes]:
    """Read a /static/... or /samples/... or uploads path from disk."""
    if not path:
        return None
    rel = path.lstrip("/")
    for base, prefix in ((STATIC_DIR, "static/"), (SAMPLES_DIR, "samples/"), (UPLOAD_FOLDER, "uploads/")):
        if rel.startswith(prefix):
            full = os.path.normpath(os.path.join(base, rel[len(prefix):]))
            if full.startswith(os.path.normpath(base)) and os.path.isfile(full):
                with open(full, "rb") as f:
                    return f.read()
    return None


def sync_catalog_to_ai(force: bool = False):
    """Push catalog images into the FashionCLIP service (it keeps vectors in memory)."""
    if not ai.online("fashion"):
        return
    with _sync_lock:
        if _sync_state["running"]:
            return
        _sync_state["running"] = True
    try:
        with app.app_context():
            items = Clothing.query.filter(Clothing.sku.isnot(None)).all()
            if not force and ai.catalog_size() == len(items):
                return
            ai.catalog_reset()  # the service keeps vectors in memory; never leave stale ids behind
            _sync_state.update(done=0, total=len(items))
            for c in items:
                img = _image_bytes_for(c.image_path)
                if img:
                    meta = ai.catalog_add(str(c.id), c.name, c.category, img)
                    if meta and not c.fabric and meta.get("fabric"):
                        c.fabric = meta["fabric"]
                _sync_state["done"] += 1
            db.session.commit()
            _sync_state["last"] = time.time()
    finally:
        _sync_state["running"] = False


def ensure_catalog_synced() -> bool:
    """True when the AI catalog is ready; otherwise kicks off a background sync."""
    if not ai.online("fashion"):
        return False
    with app.app_context():
        n = Clothing.query.filter(Clothing.sku.isnot(None)).count()
    if ai.catalog_size() == n and not _sync_state["running"]:
        return True
    threading.Thread(target=sync_catalog_to_ai, daemon=True).start()
    return False


# =========================================================
# ANALYSIS PIPELINE
# =========================================================
# Median stature (cm) by age, used only when the user gives no height and no marker is visible.
_MEDIAN_HEIGHT_CHILD = [76, 87, 96, 103, 110, 116, 122, 128, 133, 138, 144, 150, 157, 163]


def median_height(age: Optional[float], gender: str) -> float:
    if age is not None and age < 14:
        return float(_MEDIAN_HEIGHT_CHILD[max(0, min(13, int(round(age)) - 1))])
    return 162.0 if gender == "female" else 175.0 if gender == "male" else 169.0


def run_ai_fashion_analysis(image_bytes: bytes, user_height_cm: Optional[float] = None,
                            gender: Optional[str] = None, marker_cm: Optional[float] = None) -> Dict[str, Any]:
    """Age/gender -> height scale -> body measurements -> size -> catalog recommendations."""
    services_used: List[str] = []
    notes: List[str] = []

    # 1. age & gender (MiVOLO)
    age = age_range = face_box = gender_detected = gender_p = None
    age_res = ai.estimate_age(image_bytes)
    if age_res and age_res.get("mivolo"):
        services_used.append("age")
        age = age_res["mivolo"]["age"]
        age_range = age_res.get("age_range")
        face_box = age_res.get("face_box")
        gender_detected = age_res["mivolo"]["gender"]
        gender_p = age_res["mivolo"]["gender_p"]
    user_gender = (gender or "").lower()
    if user_gender in ("male", "female"):
        gender_eff, gender_source = user_gender, "user"
    elif gender_detected and (gender_p or 0) >= 0.7:
        gender_eff, gender_source = gender_detected, "detected"
    else:
        gender_eff, gender_source = "unisex", "unknown"

    # 2. height scale: printed marker > user-entered > estimate from age/gender
    # The service always looks for an ArUco marker (side length marker_cm, default 15 cm).
    height_res = ai.estimate_height(image_bytes, user_height_cm, marker_cm)
    marker_h = None
    full_body = None
    if height_res and "_error" not in height_res:
        services_used.append("height")
        full_body = height_res.get("full_body_visible")
        marker_h = (height_res.get("methods", {}).get("aruco") or {}).get("height_cm")
    elif height_res and height_res.get("_error") == 422:
        notes.append("Height service could not find a person in the photo.")
    if marker_h:
        height, height_source = float(marker_h), "marker"
        if user_height_cm and abs(user_height_cm - marker_h) > 8:
            notes.append(f"Marker says {marker_h:.0f} cm but you entered {user_height_cm:.0f} cm - using the marker.")
    elif user_height_cm:
        height, height_source = float(user_height_cm), "user"
    else:
        height, height_source = median_height(age, gender_eff), "estimated"
        notes.append("No height given - measurements are scaled to a typical height for the detected age/gender. "
                     "Enter your height or include a printed marker for accurate sizes.")
    if full_body is False:
        notes.append("Head or feet are cut off in the photo - full-body photos give better measurements.")

    # 3. body measurements + face/skin analysis (MediaPipe locally)
    cv = analyze_image_bytes(image_bytes, user_height_cm=height,
                             gender=gender_eff if gender_eff != "unisex" else "unisex",
                             age=age, height_source=height_source, face_box=face_box)
    measurements = {**cv["body_measurements"], "height_cm": round(height, 1)}
    if full_body is not None and cv.get("analysis_method") == "mediapipe_real":
        # the segmentation-based check from the height service is stricter than the landmark heuristic
        measurements["full_body_visible"] = bool(full_body)
        cv["measurement_confidence"] = measurement_confidence(cv["analysis_method"], measurements, height_source)
        cv["fit_confidence"] = cv["measurement_confidence"]["score"]
    if height_res and "_error" not in height_res and height_res.get("measurements_cm"):
        # limb lengths from the heavier pose model; already at the marker / user-height scale
        extra = height_res["measurements_cm"]
        measurements["inseam_cm"] = extra.get("inseam_approx")
        measurements["arm_cm"] = extra.get("arm_length", measurements.get("arm_cm"))

    child = is_child(age)
    # one size per garment type, from the same fit engine the product cards use (a person can be
    # an M in tops but an S in trousers)
    aud = ("girls" if gender_eff == "female" else "boys") if child else ("women" if gender_eff == "female" else "men")
    size_cats = ["tops", "bottoms"] + (["dresses"] if gender_eff == "female" else [])
    sizes_by_category = {}
    for cat in size_cats:
        rep = fit_report(measurements, cat, aud, is_child=child)
        if rep.get("supported"):
            sizes_by_category[cat] = rep["recommended_size"]
    if gender_source == "unknown" and not child:
        notes.append("Gender could not be detected - choose it above so recommendations and sizes match you.")
    analysis = {
        "success": True,
        "body_type": cv["body_type"],
        "face_shape": cv["face_shape"],
        "face_analysis": cv.get("face_analysis"),
        "recommended_frames": (cv.get("face_analysis") or {}).get("frame_advice", []),
        "skin_tone": cv["skin_tone"],
        "undertone": cv["undertone"],
        "clothing_size": sizes_by_category.get("tops", cv["clothing_size"]),
        "sizes_by_category": sizes_by_category,
        "size_system": cv["size_system"],
        "recommended_colors": cv["recommended_colors"],
        "age": round(age, 1) if age is not None else None,
        "age_range": age_range,
        "is_child": child,
        "gender": gender_eff,
        "gender_source": gender_source,
        "height_cm": round(height, 1),
        "height_source": height_source,
        "fit_confidence": cv["fit_confidence"],
        "measurement_confidence": cv["measurement_confidence"],
        "pose_detected": bool(cv.get("landmarks_detected")),
        "face_detected": face_box is not None,
        "analysis_engine": cv.get("analysis_method"),
        "analysis_method": cv.get("analysis_method"),
        "landmarks_detected": cv.get("landmarks_detected"),
        "services_used": services_used,
        "notes": notes,
        "body_measurements": measurements,
    }

    # 4. recommendations from the real catalog (AI-ranked when available)
    recs = recommend_for_profile(analysis)
    analysis["catalog_recommendations"] = recs["groups"]
    analysis["recommendation_engine"] = recs["engine"]
    names = lambda cats: [i["name"] for c in cats for i in recs["groups"].get(c, [])]  # noqa: E731
    analysis["recommended_clothes"] = names(["tops", "outerwear", "dresses", "bottoms"])[:6] or cv["recommended_clothes"]
    analysis["recommended_shoes"] = names(["footwear"])[:4] or cv["recommended_shoes"]
    analysis["recommended_sunglasses"] = names(["accessories"])[:4] or cv["recommended_sunglasses"]
    return analysis


# =========================================================
# RECOMMENDATION ENGINE
# =========================================================
def _audiences(profile: Dict[str, Any]) -> List[str]:
    g = profile.get("gender")
    if profile.get("is_child"):
        return {"male": ["boys"], "female": ["girls"]}.get(g, ["boys", "girls"])
    return {"male": ["men", "unisex"], "female": ["women", "unisex"]}.get(g, ["men", "women", "unisex"])


def size_advice(c: "Clothing", profile: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Which size of this product fits the analysed body, and is it in stock?"""
    m = profile.get("body_measurements")
    if not m or c.category not in ("tops", "outerwear", "bottoms", "dresses"):
        return None
    rep = fit_report(m, c.category, c.audience or "unisex", is_child=bool(profile.get("is_child")),
                     chart=_json(c.size_chart, {}) or None, available=_json(c.available_sizes, []) or None,
                     fit_type=c.fit_type)
    if not rep.get("supported"):
        return None
    return {"size": rep["recommended_size"], "in_stock": rep["recommended_available"], "suggested": rep["closest_available"]}


def recommend_for_profile(profile: Dict[str, Any], occasion: str = "everyday", liked_ids: Optional[List[int]] = None,
                          per_group: int = 4, max_price: Optional[float] = None,
                          only_my_size: bool = False) -> Dict[str, Any]:
    liked_ids = [int(i) for i in (liked_ids or []) if str(i).isdigit()]
    audiences = _audiences(profile)
    items = Clothing.query.filter(Clothing.audience.in_(audiences)).all()
    if not items:
        items = Clothing.query.all()
    palette = PALETTE_COLOURS.get(profile.get("undertone") or "Neutral", PALETTE_COLOURS["Neutral"])
    bt = (profile.get("body_type") or "").lower()
    boost_types = next((v for k, v in BODY_TYPE_BOOST.items() if k in bt), set())
    occ_text = OCCASIONS.get(occasion, occasion or OCCASIONS["everyday"])
    if profile.get("is_child"):
        occ_text = "kids clothing, " + occ_text

    ai_scores: Dict[int, float] = {}
    engine = "rules"
    if ensure_catalog_synced():
        res = ai.recommend([str(i) for i in liked_ids], occasion=occ_text,
                           palette=[p.lower() for p in palette], k=400)
        if res:
            engine = "fashionclip"
            ai_scores = {int(r["id"]): float(r["score"]) for r in res if str(r.get("id", "")).isdigit()}

    occ_usage = {"office": "Formal", "party": "Party", "sport": "Sports", "ethnic": "Ethnic"}.get(occasion, "Casual")
    scored = []
    filtered = {"budget": 0, "size": 0}
    for c in items:
        if c.id in liked_ids:
            continue
        if max_price is not None and c.price is not None and c.price > max_price:
            filtered["budget"] += 1
            continue
        advice = size_advice(c, profile)
        if only_my_size and advice and not advice["in_stock"]:
            filtered["size"] += 1
            continue
        reasons = []
        if advice:
            reasons.append(f"your size {advice['size']} in stock" if advice["in_stock"]
                           else f"your size {advice['size']} sold out")
        s = ai_scores.get(c.id, 0.0) if engine == "fashionclip" else 0.0
        if engine == "fashionclip" and c.id in ai_scores:
            reasons.append("AI style match")
        if c.colour_name in palette:
            s += 0.04
            reasons.append(f"{c.colour_name} suits your {profile.get('undertone', 'Neutral').lower()} undertone")
        if c.article_type in boost_types:
            s += 0.03
            reasons.append(f"flatters a {profile.get('body_type', '').split('(')[0].strip().lower()} build")
        if (c.usage or "Casual") == occ_usage:
            s += 0.03
            reasons.append(f"{occ_usage.lower()} wear")
        name_l = (c.name or "").lower()
        age = profile.get("age")
        if age is not None and age >= 4 and ("infant" in name_l or "baby" in name_l):
            s -= 0.2  # baby/infant sizes don't fit older kids
        if age is not None and age < 10 and "teen" in name_l:
            s -= 0.1
        scored.append((s, c, reasons, advice))
    scored.sort(key=lambda t: -t[0])

    groups: Dict[str, List[Dict]] = {}
    for s, c, reasons, advice in scored:
        g = groups.setdefault(c.category, [])
        if len(g) < per_group and not any(x["article_type"] == c.article_type and x["colour_name"] == c.colour_name
                                          for x in g):  # keep each group varied
            g.append({**clothing_dict(c, s, reasons), "size_advice": advice})
    return {"engine": engine, "occasion": occasion, "audiences": audiences, "filtered_out": filtered,
            "max_price": max_price, "only_my_size": only_my_size,
            "groups": {k: groups[k] for k in CATEGORY_GROUPS if k in groups}}


# =========================================================
# REST API
# =========================================================
@app.route("/", methods=["GET"])
def home():
    accept = request.headers.get("Accept", "")
    if "text/html" in accept and request.args.get("format") != "json" and os.path.exists(COMBINED_HTML):
        return send_file(COMBINED_HTML)
    return jsonify({"message": "StyleMate Backend is running", "database": "SQLite", "status": "success"})


@app.route("/api/status", methods=["GET"])
def api_status():
    return jsonify({
        "message": "StyleMate Backend is running",
        "database": "SQLite",
        "status": "ONLINE",
        "users_count": User.query.count(),
        "analyses_count": Analysis.query.count(),
        "clothing_count": Clothing.query.count(),
        "saved_outfits_count": SavedOutfit.query.count(),
    })


@app.route("/api/services", methods=["GET"])
def api_services():
    st = ai.status()
    return jsonify({"services": st, "catalog_sync": dict(_sync_state),
                    "disabled": ai.disabled()})


# ---------------------------------------------------------------- users
@app.route("/api/users", methods=["POST"])
def add_user():
    data = request.get_json(silent=True) or {}
    name, email = (data.get("name") or "").strip(), (data.get("email") or "").strip()
    if not name or not email:
        return jsonify({"error": "Name and email are required"}), 400
    existing = User.query.filter_by(email=email).first()
    if existing:
        return jsonify({"message": "User already registered", "user_id": existing.id,
                        "name": existing.name, "email": existing.email}), 200
    user = User(name=name, email=email)
    db.session.add(user)
    db.session.commit()
    return jsonify({"message": "User created", "user_id": user.id, "name": user.name, "email": user.email}), 201


@app.route("/api/users", methods=["GET"])
def list_users():
    return jsonify([{"id": u.id, "name": u.name, "email": u.email} for u in User.query.all()])


# ---------------------------------------------------------------- analysis
def _float_or_none(v) -> Optional[float]:
    try:
        f = float(v)
        return f if f > 0 else None
    except (TypeError, ValueError):
        return None


def _analyze_request(strict_user: bool):
    """Shared body of /api/upload (strict user check) and /api/analyze (lenient)."""
    image_bytes, filename = b"", "upload.jpg"
    if "image" in request.files:
        f = request.files["image"]
        if strict_user and not f.filename:
            return jsonify({"error": "No file selected"}), 400
        filename = secure_filename(f.filename or "") or filename
        image_bytes = f.read()
    elif not strict_user:
        image_bytes = request.get_data()
    else:
        return jsonify({"error": "Please attach an image"}), 400
    if not image_bytes:
        return jsonify({"error": "Uploaded image is empty" if strict_user else "Please attach an image"}), 400

    ext = os.path.splitext(filename)[1].lower() or ".jpg"
    if ext not in ALLOWED_EXT:
        return jsonify({"error": "Only JPG, JPEG, PNG and WEBP files are allowed"}), 400

    user_id = request.form.get("user_id", "1")
    if strict_user:
        if not str(user_id).isdigit():
            return jsonify({"error": "Valid user_id is required"}), 400
        user = db.session.get(User, int(user_id))
        if not user:
            return jsonify({"error": "User not found"}), 404
    else:
        user = db.session.get(User, int(user_id)) if str(user_id).isdigit() else None
        user = user or User.query.first()
    user_id = user.id

    unique_name = f"{uuid4()}_{filename}"
    with open(os.path.join(UPLOAD_FOLDER, unique_name), "wb") as fh:
        fh.write(image_bytes)

    result = run_ai_fashion_analysis(
        image_bytes,
        user_height_cm=_float_or_none(request.form.get("height_cm")),
        gender=request.form.get("gender"),
        marker_cm=_float_or_none(request.form.get("marker_cm")),
    )
    analysis = Analysis(
        user_id=user_id, image_path=unique_name, face_shape=result["face_shape"],
        body_measurements=json.dumps({**result["body_measurements"],
                                      "recommended_clothes": result["recommended_clothes"],
                                      "recommended_shoes": result["recommended_shoes"],
                                      "recommended_sunglasses": result["recommended_sunglasses"]}),
        age=result["age"], gender=result["gender"], height_cm=result["height_cm"],
        height_source=result["height_source"], result_json=json.dumps(result),
    )
    db.session.add(analysis)
    db.session.commit()
    return jsonify({"message": "Image uploaded and analyzed successfully", "status": "completed",
                    "analysis_id": analysis.id, "user_id": user_id, "image_path": unique_name,
                    "image_url": f"/uploads/{unique_name}", **result}), (201 if strict_user else 200)


@app.route("/api/upload", methods=["POST"])
def upload_image():
    return _analyze_request(strict_user=True)


@app.route("/api/analyze", methods=["POST"])
def analyze_and_store():
    return _analyze_request(strict_user=False)


@app.route("/api/analysis", methods=["POST"])
def save_analysis():
    data = request.get_json(silent=True) or {}
    analysis_id = data.get("analysis_id")
    if analysis_id is None:
        return jsonify({"error": "analysis_id is required"}), 400
    analysis = db.session.get(Analysis, analysis_id)
    if analysis is None:
        return jsonify({"error": "Analysis record not found"}), 404
    if data.get("body_measurements") is not None:
        analysis.body_measurements = json.dumps(data["body_measurements"])
    if data.get("face_shape") is not None:
        analysis.face_shape = data["face_shape"]
    db.session.commit()
    return jsonify({"message": "Analysis results saved", "analysis_id": analysis.id})


@app.route("/api/analysis/<int:analysis_id>", methods=["GET"])
def get_analysis(analysis_id):
    a = db.session.get(Analysis, analysis_id)
    if a is None:
        return jsonify({"error": "Analysis not found"}), 404
    return jsonify({"analysis_id": a.id, "user_id": a.user_id, "image_path": a.image_path,
                    "body_measurements": json.loads(a.body_measurements) if a.body_measurements else None,
                    "face_shape": a.face_shape, "age": a.age, "gender": a.gender, "height_cm": a.height_cm,
                    "height_source": a.height_source})


def _profile_for(analysis_id) -> Dict[str, Any]:
    a = db.session.get(Analysis, int(analysis_id)) if str(analysis_id or "").isdigit() else None
    if a and a.result_json:
        return json.loads(a.result_json)
    return {"gender": "unisex", "undertone": "Neutral", "body_type": ""}


@app.route("/api/recommendations", methods=["POST"])
def recommendations():
    data = request.get_json(silent=True) or {}
    profile = _profile_for(data.get("analysis_id"))
    if data.get("gender") in ("male", "female"):
        profile["gender"] = data["gender"]
    max_price = data.get("max_price")
    if max_price not in (None, ""):
        try:
            max_price = float(max_price)
            if max_price < 0:
                raise ValueError
        except (TypeError, ValueError):
            return jsonify({"error": "max_price must be a non-negative number"}), 400
    else:
        max_price = None
    res = recommend_for_profile(profile, occasion=data.get("occasion") or "everyday",
                                liked_ids=data.get("liked_ids") or [], per_group=int(data.get("per_group", 4)),
                                max_price=max_price, only_my_size=bool(data.get("only_my_size", False)))
    return jsonify(res)


# ---------------------------------------------------------------- catalog
PRODUCT_TEXT_FIELDS = ("name", "category", "color", "image_path", "model_path", "audience", "article_type",
                       "colour_name", "usage", "fabric", "brand", "currency", "fit_type", "pattern")
FIT_TYPES = ("slim", "skinny", "regular", "relaxed", "loose", "oversized")


def apply_product_fields(item: "Clothing", data: Dict[str, Any]) -> Optional[str]:
    """Validate and copy catalog fields from JSON onto a Clothing row. Returns an error message or None."""
    for f in PRODUCT_TEXT_FIELDS:
        if f in data:
            v = data[f]
            if v is not None and not isinstance(v, str):
                return f"{f} must be text"
            setattr(item, f, v.strip() if isinstance(v, str) else v)
    if "price" in data:
        try:
            item.price = None if data["price"] in (None, "") else float(data["price"])
        except (TypeError, ValueError):
            return "price must be a number"
        if item.price is not None and item.price < 0:
            return "price must not be negative"
        item.price_is_demo = bool(data.get("price_is_demo", False))
    if "available_sizes" in data:
        sizes = data["available_sizes"]
        if not isinstance(sizes, list) or not all(isinstance(x, str) and x.strip() for x in sizes):
            return "available_sizes must be a list of size labels"
        item.available_sizes = json.dumps([x.strip() for x in sizes])
    if "size_chart" in data:
        chart = data["size_chart"]
        ok = isinstance(chart, dict) and all(
            isinstance(v, dict) and all(k in ("chest", "waist", "hip") and isinstance(n, (int, float)) and n > 0
                                        for k, n in v.items()) for v in chart.values())
        if not ok:
            return "size_chart must be {size: {chest|waist|hip: positive cm}}"
        item.size_chart = json.dumps(chart)
    if item.fit_type and item.fit_type.lower() not in FIT_TYPES:
        return "fit_type must be one of: " + ", ".join(FIT_TYPES)
    return None


@app.route("/api/clothing", methods=["POST"])
def add_clothing():
    data = request.get_json(silent=True) or {}
    name, category = data.get("name"), data.get("category")
    if not name or not category:
        return jsonify({"error": "Name and category are required"}), 400
    item = Clothing(name=name, category=category, audience="unisex")
    err = apply_product_fields(item, data)
    if err:
        return jsonify({"error": err}), 400
    item.fabric = item.fabric or guess_fabric(name, item.article_type)
    db.session.add(item)
    db.session.commit()
    return jsonify({"message": "Clothing item added", "item_id": item.id}), 201


@app.route("/api/clothing/<int:item_id>", methods=["PATCH"])
def update_clothing(item_id):
    item = db.session.get(Clothing, item_id)
    if not item:
        return jsonify({"error": "Clothing item not found"}), 404
    data = request.get_json(silent=True) or {}
    if not data:
        return jsonify({"error": "Send the fields to update as JSON"}), 400
    err = apply_product_fields(item, data)
    if err:
        db.session.rollback()
        return jsonify({"error": err}), 400
    db.session.commit()
    return jsonify({"message": "Clothing item updated", "item": clothing_dict(item, with_chart=True)})


@app.route("/api/clothing", methods=["GET"])
def get_clothing():
    q = Clothing.query
    if request.args.get("category"):
        q = q.filter_by(category=request.args["category"])
    if request.args.get("audience"):
        q = q.filter(Clothing.audience.in_(request.args["audience"].split(",")))
    if request.args.get("brand"):
        q = q.filter(Clothing.brand.ilike(request.args["brand"]))
    if request.args.get("source") == "user":  # garments added through the clothing upload, newest first
        q = q.filter(Clothing.sku.like("user-%")).order_by(Clothing.id.desc())
    if request.args.get("max_price"):
        try:
            q = q.filter(Clothing.price <= float(request.args["max_price"]))
        except ValueError:
            return jsonify({"error": "max_price must be a number"}), 400
    items = [clothing_dict(c) for c in q.all()]
    if request.args.get("size"):
        items = [i for i in items if request.args["size"] in i["available_sizes"]]
    return jsonify(items)


@app.route("/api/clothing/<int:item_id>", methods=["GET"])
def get_clothing_item(item_id):
    c = db.session.get(Clothing, item_id)
    return (jsonify(clothing_dict(c, with_chart=True)), 200) if c else (jsonify({"error": "Clothing item not found"}), 404)


@app.route("/api/search", methods=["GET"])
def search_catalog():
    q = (request.args.get("q") or "").strip()
    k = min(int(request.args.get("k", 12)), 48)
    audiences = [a for a in (request.args.get("audience") or "").split(",") if a]
    if not q:
        return jsonify({"engine": "none", "results": []})
    results, engine = [], "keyword"
    if ensure_catalog_synced():
        hits = ai.search(q, k=k * 3) or []
        for h in hits:
            c = db.session.get(Clothing, int(h["id"])) if str(h.get("id", "")).isdigit() else None
            if c and (not audiences or c.audience in audiences):
                results.append(clothing_dict(c, h["score"]))
        engine = "fashionclip"
    if not results:
        engine = "keyword"
        terms = q.lower().split()
        for c in Clothing.query.all():
            hay = " ".join(filter(None, [c.name, c.article_type, c.colour_name, c.usage, c.fabric, c.category])).lower()
            score = sum(t in hay for t in terms) / len(terms)
            if score > 0 and (not audiences or c.audience in audiences):
                results.append(clothing_dict(c, score))
        results.sort(key=lambda r: -r["score"])
    seen, unique = set(), []
    for r in results:  # the dataset has a few products that share a display name
        if r["name"] not in seen:
            seen.add(r["name"])
            unique.append(r)
    return jsonify({"engine": engine, "results": unique[:k]})


@app.route("/api/clothing/<int:item_id>/similar", methods=["GET"])
def similar_items(item_id):
    base = db.session.get(Clothing, item_id)
    if not base:
        return jsonify({"error": "Clothing item not found"}), 404
    k = min(int(request.args.get("k", 8)), 24)
    if ensure_catalog_synced():
        hits = ai.similar(str(item_id), k=k * 3) or []
        out = []
        for h in hits:
            c = db.session.get(Clothing, int(h["id"])) if str(h.get("id", "")).isdigit() else None
            if c and c.audience in (base.audience, "unisex"):
                out.append(clothing_dict(c, h["score"]))
        if out:
            return jsonify({"engine": "fashionclip", "results": out[:k]})
    same = (Clothing.query.filter(Clothing.id != item_id, Clothing.article_type == base.article_type,
                                  Clothing.audience == base.audience).limit(k).all())
    return jsonify({"engine": "rules", "results": [clothing_dict(c) for c in same]})


@app.route("/api/clothing/<int:item_id>/complete", methods=["GET"])
def complete_outfit(item_id):
    """Items from the other categories that go with this one."""
    base = db.session.get(Clothing, item_id)
    if not base:
        return jsonify({"error": "Clothing item not found"}), 404
    profile = {"gender": {"men": "male", "women": "female", "boys": "male", "girls": "female"}.get(base.audience, "unisex"),
               "is_child": base.audience in ("boys", "girls"), "undertone": "Neutral", "body_type": ""}
    res = recommend_for_profile(profile, occasion=(base.usage or "everyday").lower(), liked_ids=[item_id], per_group=3)
    res["groups"] = {k: v for k, v in res["groups"].items() if k != base.category}
    return jsonify(res)


# ---------------------------------------------------------------- clothing upload (cloth analysis)
# Garment types a user upload can become: key -> (label, catalog category, article_type, VisionFit label).
# article_type is what the 3D avatar reads to pick a cut (sleeves, skirt, shorts...); the VisionFit label
# is the class of Akshit Kandari's classifier (services/cloth_service.py) that covers this type.
GARMENT_TYPES = {
    "tshirt": ("T-shirt", "tops", "Tshirts", "t-shirt"),
    "shirt": ("Shirt", "tops", "Shirts", "shirt"),
    "longsleeve": ("Long-sleeve top", "tops", "Long Sleeve", "longsleeve"),
    "sweater": ("Sweater / hoodie", "tops", "Sweatshirts", "longsleeve"),
    "kurta": ("Kurta", "tops", "Kurtas", None),
    "jacket": ("Jacket / coat", "outerwear", "Jackets", "outwear"),
    "blazer": ("Blazer", "outerwear", "Blazers", "outwear"),
    "dress": ("Dress", "dresses", "Dresses", "dress"),
    "saree": ("Saree", "dresses", "Sarees", None),
    "skirt": ("Skirt", "bottoms", "Skirts", "skirt"),
    "jeans": ("Jeans", "bottoms", "Jeans", "pants"),
    "trousers": ("Trousers", "bottoms", "Trousers", "pants"),
    "trackpants": ("Track pants", "bottoms", "Track Pants", "pants"),
    "shorts": ("Shorts", "bottoms", "Shorts", "shorts"),
    "shoes": ("Shoes / sneakers", "footwear", "Casual Shoes", "shoes"),
    "formalshoes": ("Formal shoes", "footwear", "Formal Shoes", "shoes"),
    "sandals": ("Sandals", "footwear", "Sandals", "shoes"),
    "heels": ("Heels", "footwear", "Heels", "shoes"),
    "cap": ("Cap / hat", "accessories", "Caps", "hat"),
    "sunglasses": ("Sunglasses", "accessories", "Sunglasses", None),
    "watch": ("Watch", "accessories", "Watches", None),
    "handbag": ("Handbag", "accessories", "Handbags", None),
    "backpack": ("Backpack", "accessories", "Backpacks", None),
    "belt": ("Belt", "accessories", "Belts", None),
}
VISIONFIT_TO_TYPE = {"dress": "dress", "hat": "cap", "longsleeve": "longsleeve", "outwear": "jacket", "pants": "trousers",
                     "shirt": "shirt", "shoes": "shoes", "shorts": "shorts", "skirt": "skirt", "t-shirt": "tshirt"}
FASHIONCLIP_TO_TYPE = {"t-shirt": "tshirt", "shirt": "shirt", "polo shirt": "tshirt", "sweater": "sweater",
                       "hoodie": "sweater", "jacket": "jacket", "blazer": "blazer", "coat": "jacket", "dress": "dress",
                       "skirt": "skirt", "jeans": "jeans", "trousers": "trousers", "shorts": "shorts",
                       "track pants": "trackpants", "kurta": "kurta", "saree": "saree", "sneakers": "shoes",
                       "formal shoes": "formalshoes", "sandals": "sandals", "heels": "heels", "handbag": "handbag",
                       "watch": "watch", "sunglasses": "sunglasses", "cap": "cap"}
# fabric presets the 3D cloth knows (static/js/avatar_studio.js FABRICS, same list as FashionCLIP's)
GARMENT_FABRICS = ("cotton", "denim", "linen", "silk", "satin", "wool", "knit", "leather", "suede", "polyester",
                   "chiffon", "velvet", "fleece", "corduroy", "lace", "nylon")
COLOUR_NAMES = {"Black": (20, 20, 22), "White": (240, 240, 238), "Grey": (128, 128, 130), "Charcoal": (62, 62, 64),
                "Navy Blue": (30, 42, 105),
                "Blue": (40, 90, 190), "Red": (200, 30, 40), "Maroon": (120, 25, 40), "Pink": (235, 140, 175),
                "Purple": (110, 60, 150), "Green": (40, 130, 70), "Olive": (110, 115, 50), "Yellow": (235, 200, 50),
                "Orange": (235, 120, 40), "Brown": (110, 70, 40), "Beige": (215, 195, 160), "Cream": (240, 230, 205)}


def garment_colour(data: bytes) -> Optional[Dict[str, str]]:
    """Main colour of the garment in a photo: drop pixels close to the border (background) colour, then take
    the most common of 6 quantised colours. Returns {hex, name} or None if the image can't be read."""
    from PIL import Image, ImageOps
    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    except Exception:  # noqa: BLE001
        return None
    img.thumbnail((96, 96))
    w, h = img.size
    raw = img.tobytes()
    px = [tuple(raw[i:i + 3]) for i in range(0, len(raw), 3)]
    border = [px[i] for i in range(w)] + [px[(h - 1) * w + i] for i in range(w)] + \
             [px[j * w] for j in range(h)] + [px[j * w + w - 1] for j in range(h)]
    bg = tuple(sorted(c[k] for c in border)[len(border) // 2] for k in range(3))
    fg = [c for c in px if sum((c[k] - bg[k]) ** 2 for k in range(3)) > 45 ** 2]
    if len(fg) < len(px) * 0.05:  # garment fills the frame or matches the background: use the centre
        fg = [px[j * w + i] for j in range(h // 4, 3 * h // 4) for i in range(w // 4, 3 * w // 4)]
    q = Image.frombytes("RGB", (len(fg), 1), bytes(v for c in fg for v in c)).quantize(colors=6)
    count, idx = max(q.getcolors())
    rgb = tuple(q.getpalette()[idx * 3: idx * 3 + 3])
    name = min(COLOUR_NAMES, key=lambda n: sum((rgb[k] - COLOUR_NAMES[n][k]) ** 2 for k in range(3)))
    return {"hex": "#%02x%02x%02x" % rgb, "name": name}


def analyze_garment(data: bytes) -> Dict[str, Any]:
    """Run both garment models and propose catalog fields. The VisionFit classifier and FashionCLIP each vote
    for a type; FashionCLIP's finer type is used when both agree (or VisionFit is offline). On our catalog
    FashionCLIP was right 94% of the time vs 55% for VisionFit, so on disagreement it is the default and the
    upload is flagged for the user to check."""
    vf = ai.classify_cloth(data)
    fc = ai.tag_garment(data)
    colour = garment_colour(data) or {"hex": "#888888", "name": "Grey"}
    vf_type = VISIONFIT_TO_TYPE.get(vf["category"]) if vf else None
    fc_type = FASHIONCLIP_TO_TYPE.get(fc["type"][0]["label"]) if fc else None
    agree = None
    if vf and fc_type:
        agree = GARMENT_TYPES[fc_type][3] == vf["category"]
    gtype = fc_type or vf_type or "tshirt"
    fabric = fc["fabric"][0]["label"] if fc else None
    if fabric not in GARMENT_FABRICS:
        fabric = guess_fabric(GARMENT_TYPES[gtype][0], GARMENT_TYPES[gtype][2]) or "cotton"
    usage = fc["style"][0]["label"].title() if fc else "Casual"
    colour_name = fc["colour"][0]["label"].title() if fc else colour["name"]
    return {
        "visionfit": vf, "fashionclip": fc, "colour": colour, "agree": agree,
        "suggested": {"type": gtype, "name": f"My {colour_name.lower()} {GARMENT_TYPES[gtype][0].split(' /')[0].lower()}",
                      "color": colour["hex"], "colour_name": colour_name, "fabric": fabric, "usage": usage},
        "engines": [e for e, r in (("visionfit", vf), ("fashionclip", fc)) if r],
    }


def create_user_garment(data: bytes, filename: str, gtype: str, name: str, color: str, colour_name: str,
                        fabric: str, usage: str, audience: str, price: Optional[float] = None) -> "Clothing":
    label, category, article_type, _ = GARMENT_TYPES[gtype]
    fname = f"{uuid4().hex}{os.path.splitext(secure_filename(filename))[1].lower() or '.jpg'}"
    with open(os.path.join(USER_GARMENTS_DIR, fname), "wb") as fh:
        fh.write(data)
    item = Clothing(name=name, category=category, color=color, image_path=f"/static/catalog/user/{fname}",
                    model_path=f"3d://mannequin/{category}", sku=f"user-{fname[:12]}", article_type=article_type,
                    audience=audience, colour_name=colour_name, usage=usage, fabric=fabric,
                    price=price, price_is_demo=False, currency="INR", fit_type="regular")
    db.session.add(item)
    db.session.commit()
    ai.catalog_add(str(item.id), item.name, item.category, data)
    return item


def _uploaded_image():
    f = request.files.get("image")
    if not f or not f.filename:
        return None, None, (jsonify({"error": "Please attach a photo of the garment"}), 400)
    data = f.read()
    if not data:
        return None, None, (jsonify({"error": "Uploaded image is empty"}), 400)
    if len(data) > 10 * 1024 * 1024:
        return None, None, (jsonify({"error": "That image is over 10 MB"}), 400)
    if garment_colour(data) is None:
        return None, None, (jsonify({"error": "That file is not an image we can read"}), 400)
    return data, f.filename, None


@app.route("/api/clothing/analyze", methods=["POST"])
def analyze_clothing():
    """Garment photo -> cloth analysis (VisionFit classifier + FashionCLIP + pixel colour) and suggested fields."""
    data, _, err = _uploaded_image()
    if err:
        return err
    out = analyze_garment(data)
    out["types"] = [{"key": k, "label": v[0], "category": v[1]} for k, v in GARMENT_TYPES.items()]
    out["fabrics"] = list(GARMENT_FABRICS)
    return jsonify(out)


@app.route("/api/clothing/upload", methods=["POST"])
def upload_clothing():
    """Add an analysed garment photo to the catalog with the fields the user confirmed."""
    data, filename, err = _uploaded_image()
    if err:
        return err
    form = request.form
    gtype = form.get("type", "")
    if gtype not in GARMENT_TYPES:
        return jsonify({"error": "Choose what kind of garment this is"}), 400
    name = (form.get("name") or "").strip()[:200]
    if not name:
        return jsonify({"error": "Give the garment a name"}), 400
    color = (form.get("color") or "").strip().lower()
    if not (len(color) == 7 and color[0] == "#" and all(ch in "0123456789abcdef" for ch in color[1:])):
        return jsonify({"error": "color must be a hex colour like #1e3a8a"}), 400
    fabric = (form.get("fabric") or "cotton").lower()
    if fabric not in GARMENT_FABRICS:
        return jsonify({"error": "fabric must be one of: " + ", ".join(GARMENT_FABRICS)}), 400
    audience = form.get("audience") or "unisex"
    if audience not in ("men", "women", "unisex", "boys", "girls"):
        return jsonify({"error": "audience must be men, women, unisex, boys or girls"}), 400
    price = None
    if form.get("price"):
        try:
            price = float(form["price"])
        except ValueError:
            return jsonify({"error": "price must be a number"}), 400
        if price < 0:
            return jsonify({"error": "price must not be negative"}), 400
    item = create_user_garment(data, filename, gtype, name, color,
                               (form.get("colour_name") or "").strip()[:40] or None, fabric,
                               (form.get("usage") or "Casual").strip()[:40], audience, price)
    return jsonify({"message": "Added to the catalog", "item": clothing_dict(item)}), 201


@app.route("/api/clothing/<int:item_id>", methods=["DELETE"])
def delete_clothing(item_id):
    """Remove a user-uploaded garment (catalog items from catalog.json can't be deleted here)."""
    item = db.session.get(Clothing, item_id)
    if not item:
        return jsonify({"error": "Clothing item not found"}), 404
    if not (item.sku or "").startswith("user-"):
        return jsonify({"error": "Only clothes you uploaded can be deleted"}), 403
    SavedOutfit.query.filter_by(clothing_id=item.id).delete()
    path = os.path.join(USER_GARMENTS_DIR, os.path.basename(item.image_path or ""))
    db.session.delete(item)
    db.session.commit()
    if os.path.isfile(path):
        os.remove(path)
    return jsonify({"message": "Deleted", "item_id": item_id})  # FashionCLIP re-syncs on the next query


@app.route("/api/clothing/tag", methods=["POST"])
def tag_clothing():
    """Upload a garment photo -> AI tags (type/colour/fabric/style). add=1 also adds it to the catalog."""
    data, filename, err = _uploaded_image()
    if err:
        return err
    tags = ai.tag_garment(data)
    if tags is None:
        return jsonify({"error": "Garment tagging service (port 5103) is offline"}), 503
    out = {"tags": tags}
    if request.form.get("add") == "1":
        s = analyze_garment(data)["suggested"]
        item = create_user_garment(data, filename, s["type"], (request.form.get("name") or "").strip() or s["name"],
                                   s["color"], s["colour_name"], s["fabric"], s["usage"],
                                   request.form.get("audience", "unisex"))
        out["item"] = clothing_dict(item)
    return jsonify(out)


# ---------------------------------------------------------------- outfits
@app.route("/api/outfits/save", methods=["POST"])
def save_outfit():
    data = request.get_json(silent=True) or {}
    user_id = data.get("user_id", 1)
    clothing_id = data.get("clothing_id")
    clothing_name = data.get("clothing_name")
    accessory_id = data.get("accessory_id")

    if clothing_id is None and clothing_name:
        matched = Clothing.query.filter(Clothing.name.ilike(f"%{clothing_name}%")).first()
        if not matched:
            matched = Clothing(name=clothing_name, category="custom", color="#4f46e5",
                               image_path="/samples/2_garment_cloth.jpg", model_path="3d://mannequin/custom",
                               audience="unisex", fabric=guess_fabric(clothing_name, None))
            db.session.add(matched)
            db.session.commit()
        clothing_id = matched.id

    if user_id is None or clothing_id is None:
        return jsonify({"error": "user_id and clothing_id are required"}), 400
    if not str(user_id).isdigit() or not db.session.get(User, int(user_id)):
        return jsonify({"error": "User not found"}), 404
    clothing_obj = db.session.get(Clothing, int(clothing_id))
    if not clothing_obj:
        return jsonify({"error": "Clothing item not found"}), 404

    outfit = SavedOutfit(user_id=int(user_id), clothing_id=int(clothing_id), accessory_id=accessory_id)
    db.session.add(outfit)
    db.session.commit()
    return jsonify({"message": "Outfit saved successfully", "outfit_id": outfit.id, "clothing_name": clothing_obj.name,
                    "clothing_id": clothing_obj.id, "accessory_id": outfit.accessory_id}), 201


@app.route("/api/outfits/<int:user_id>", methods=["GET"])
def get_outfits(user_id):
    results = []
    for o in SavedOutfit.query.filter_by(user_id=user_id).all():
        c = db.session.get(Clothing, o.clothing_id)
        results.append({"outfit_id": o.id, "user_id": o.user_id, "clothing_id": o.clothing_id,
                        "clothing_name": c.name if c else f"Item #{o.clothing_id}",
                        "clothing_color": c.color if c else "#4f46e5",
                        "image_path": c.image_path if c else None, "category": c.category if c else None,
                        "accessory_id": o.accessory_id})
    return jsonify(results)


@app.route("/api/outfits/item/<int:outfit_id>", methods=["DELETE"])
def delete_outfit(outfit_id):
    o = db.session.get(SavedOutfit, outfit_id)
    if not o:
        return jsonify({"error": "Outfit not found"}), 404
    db.session.delete(o)
    db.session.commit()
    return jsonify({"message": "Outfit deleted", "outfit_id": outfit_id})


# ---------------------------------------------------------------- try-on
@app.route("/api/try-on", methods=["POST"])
def try_on():
    """Parameters for the 3D avatar: colour, fabric material, garment image."""
    data = request.get_json(silent=True) or {}
    clothing_id, clothing_name = data.get("clothing_id"), data.get("clothing_name")
    item = None
    if clothing_id is not None and str(clothing_id).isdigit():
        item = db.session.get(Clothing, int(clothing_id))
    if item is None and clothing_name:
        item = Clothing.query.filter(Clothing.name.ilike(f"%{clothing_name}%")).first()
    item = item or Clothing.query.first()
    if item is None:
        return jsonify({"error": "Clothing item not found"}), 404
    return jsonify({
        "message": "Items ready for 3D visualization",
        "clothing_id": item.id, "clothing_name": item.name, "clothing_color": item.color,
        "clothing_model": item.model_path, "category": item.category, "fabric": item.fabric,
        "material": FABRIC_MATERIAL.get(item.fabric or "", {"roughness": 0.8, "metalness": 0.0}),
        "texture_url": item.image_path, "accessory_id": data.get("accessory_id"),
        "article_type": item.article_type, "audience": item.audience,
    })


@app.route("/api/fit", methods=["POST"])
def garment_fit():
    """{analysis_id | measurements, clothing_id, size?} -> per-region ease/verdict + recommended size."""
    data = request.get_json(silent=True) or {}
    c = db.session.get(Clothing, int(data["clothing_id"])) if str(data.get("clothing_id", "")).isdigit() else None
    if not c:
        return jsonify({"error": "clothing_id is required"}), 400
    profile = _profile_for(data.get("analysis_id"))
    measurements = data.get("measurements") or profile.get("body_measurements")
    if not measurements:
        return jsonify({"supported": False, "reason": "analyze a photo first"}), 200
    rep = fit_report(measurements, c.category, c.audience or "unisex", data.get("size"),
                     is_child=bool(profile.get("is_child")), chart=_json(c.size_chart, {}) or None,
                     available=_json(c.available_sizes, []) or None, fit_type=c.fit_type)
    rep.update(clothing_id=c.id, clothing_name=c.name)
    return jsonify(rep)


# ---------------------------------------------------------------- static files
@app.route("/samples/<path:filename>", methods=["GET"])
def serve_samples(filename):
    return send_from_directory(SAMPLES_DIR, filename)


@app.route("/uploads/<path:filename>", methods=["GET"])
def serve_uploads(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/web_app/<path:filename>", methods=["GET"])
def serve_web_app(filename):
    return send_from_directory(WEB_APP_DIR, filename)


@app.route("/models/<path:filename>", methods=["GET"])
def serve_models(filename):
    return send_from_directory(os.path.join(WEB_APP_DIR, "models"), filename)


@app.route("/favicon.ico", methods=["GET"])
def favicon():
    return send_file(os.path.join(STATIC_DIR, "favicon.svg"), mimetype="image/svg+xml")


@app.route("/studio", methods=["GET"])
def studio():
    """The 3D studio lives on the main page; the legacy standalone page stays at /web_app/index.html."""
    return redirect("/#tryon-section")


with app.app_context():
    db.create_all()
    migrate_schema()
    seed_default_data()

if not ai.disabled():
    # full re-sync at startup: the service may hold vectors from another database with other ids
    threading.Thread(target=sync_catalog_to_ai, kwargs={"force": True}, daemon=True).start()


if __name__ == "__main__":
    print("=" * 68)
    print("  STYLEMATE FLASK + SQLITE + AI BACKEND (PORT 5000)")
    print("=" * 68)
    print(f" [OK] SQLite Database: {DB_PATH}")
    print(f" [OK] Uploads Folder:  {UPLOAD_FOLDER}")
    for name, s in ai.status().items():
        print(f" [{'OK' if s['online'] else '--'}] {s['description']:<58} {s['url']}")
    print(" [OK] Server URL:      http://127.0.0.1:5000")
    print("=" * 68)

    if os.environ.get("NO_BROWSER") != "1":
        def open_browser():
            time.sleep(1.5)
            webbrowser.open("http://127.0.0.1:5000")

        threading.Thread(target=open_browser, daemon=True).start()
    app.run(debug=False, port=5000, threaded=True)
