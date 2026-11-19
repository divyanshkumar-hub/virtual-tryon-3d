import os
import json
import struct

from vision_analyzer import analyze_image_bytes
import webbrowser
import threading
import time
from uuid import uuid4

from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INSTANCE_DIR = os.path.join(BASE_DIR, "instance")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")
WEB_APP_DIR = os.path.join(BASE_DIR, "web_app")
COMBINED_HTML = os.path.join(BASE_DIR, "combined_app.html")

os.makedirs(INSTANCE_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)

app = Flask(__name__, instance_path=INSTANCE_DIR)
CORS(app)

# Database and upload settings
DB_PATH = os.path.join(INSTANCE_DIR, "stylemate.db")
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{DB_PATH}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

db = SQLAlchemy(app)


# =========================================================
# DATABASE TABLES (StyleMate Schema from mini.py)
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


class Clothing(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    category = db.Column(db.String(50), nullable=False)
    color = db.Column(db.String(50))
    image_path = db.Column(db.String(255))
    model_path = db.Column(db.String(255))


class SavedOutfit(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    clothing_id = db.Column(
        db.Integer, db.ForeignKey("clothing.id"), nullable=False
    )
    accessory_id = db.Column(db.Integer)


# =========================================================
# AI / ML BODY & FACE ANALYSIS ENGINE
# =========================================================
def extract_image_features_pure_python(image_bytes):
    size_len = len(image_bytes)
    if size_len == 0:
        return 384, 512, 128

    step = max(1, size_len // 1000)
    sampled = image_bytes[::step]
    avg_tone = sum(sampled) / max(len(sampled), 1)

    w, h = 384, 512
    if image_bytes[:8] == b"\x89PNG\r\n\x1a\n" and size_len >= 24:
        try:
            w, h = struct.unpack(">II", image_bytes[16:24])
        except Exception:
            pass

    return w, h, avg_tone


def run_ai_fashion_analysis(image_bytes):
    """
    Runs CV-backed body & face shape analysis and produces measurements + outfit recommendations.
    Uses MediaPipe Pose/FaceMesh via vision_analyzer.py with automatic Pillow fallback.
    """
    cv = analyze_image_bytes(image_bytes)
    w = cv.get("image_width", 384)
    h = cv.get("image_height", 512)
    ratio = round(h / max(w, 1), 2)

    body_type = cv.get("body_type", "Balanced")
    face_shape = cv.get("face_shape", "Oval")
    undertone = cv.get("undertone", "Neutral")

    if "Inverted Triangle" in body_type or "Rectangle" in body_type:
        measurements = {
            "body_type": body_type,
            "shoulder_cm": 46.5,
            "chest_cm": 98.0,
            "waist_cm": 81.0,
            "hip_cm": 95.0,
            "aspect_ratio": ratio,
            "shoulder_hip_ratio": cv.get("shoulder_hip_ratio"),
        }
        clothes = [
            "Navy Tailored Double-Breasted Blazer",
            "Cyber Urban Hoodie",
            "Slim-Fit White Oxford Shirt",
            "Relaxed Straight-Leg Denim Jeans",
        ]
        shoes = [
            "Classic White Leather Sneakers",
            "Suede Chelsea Boots",
            "Chunky Streetwear Runners",
        ]
        sunglasses = [
            "Gold Aviator Sunglasses",
            "Tortoiseshell Wayfarer Frames",
            "Vintage Clubmaster Shades",
        ]
    elif "Pear" in body_type:
        measurements = {
            "body_type": body_type,
            "shoulder_cm": 45.0,
            "chest_cm": 96.0,
            "waist_cm": 80.0,
            "hip_cm": 100.0,
            "aspect_ratio": ratio,
            "shoulder_hip_ratio": cv.get("shoulder_hip_ratio"),
        }
        clothes = [
            "Navy Tailored Double-Breasted Blazer",
            "Slim-Fit White Oxford Shirt",
            "Vintage Leather Rider Jacket",
            "Tapered Chino Trousers",
        ]
        shoes = [
            "Minimalist White Low-Top Sneakers",
            "Brown Leather Penny Loafers",
            "High-Top Canvas Trainers",
        ]
        sunglasses = [
            "Round Retro Metal Sunglasses",
            "Gold Aviator Sunglasses",
            "Polarized Navigator Frames",
        ]
    else:
        measurements = {
            "body_type": body_type,
            "shoulder_cm": 47.0,
            "chest_cm": 100.0,
            "waist_cm": 82.0,
            "hip_cm": 96.0,
            "aspect_ratio": ratio,
            "shoulder_hip_ratio": cv.get("shoulder_hip_ratio"),
        }
        clothes = [
            "Cyber Urban Hoodie",
            "Vintage Leather Rider Jacket",
            "Slim-Fit White Oxford Shirt",
            "Tapered Chino Trousers",
        ]
        shoes = [
            "Classic White Leather Sneakers",
            "Brown Suede Chelsea Boots",
            "Retro Basketball High-Tops",
        ]
        sunglasses = [
            "Gold Aviator Sunglasses",
            "Angular Wayfarer Sunglasses",
            "Hexagonal Metal Frames",
        ]

    return {
        "success": True,
        "body_type": body_type,
        "face_shape": face_shape,
        "skin_tone": cv.get("skin_tone", "Medium"),
        "undertone": undertone,
        "recommended_colors": cv.get("recommended_colors", []),
        "pose_detected": cv.get("pose_detected", False),
        "face_detected": cv.get("face_detected", False),
        "analysis_engine": cv.get("analysis_engine", "pillow_fallback"),
        "body_measurements": measurements,
        "recommended_clothes": clothes,
        "recommended_shoes": shoes,
        "recommended_sunglasses": sunglasses,
        "tryon_image_url": "/samples/4_vton_synthesis_result.jpg",
    }


def seed_default_data():
    """Ensures a default user and 3D clothing catalog exist in stylemate.db."""
    if not User.query.first():
        default_user = User(name="StyleMate Demo User", email="demo@stylemate.ai")
        db.session.add(default_user)
        db.session.commit()

    if not Clothing.query.first():
        default_clothes = [
            Clothing(
                name="Navy Tailored Double-Breasted Blazer",
                category="outerwear",
                color="#1e3a8a",
                image_path="/samples/2_garment_cloth.jpg",
                model_path="3d://mannequin/blazer-navy",
            ),
            Clothing(
                name="Cyber Urban Hoodie",
                category="tops",
                color="#e11d48",
                image_path="/samples/4_vton_synthesis_result.jpg",
                model_path="3d://mannequin/hoodie-cyber",
            ),
            Clothing(
                name="Slim-Fit White Oxford Shirt",
                category="tops",
                color="#f8fafc",
                image_path="/samples/2_garment_cloth.jpg",
                model_path="3d://mannequin/oxford-white",
            ),
            Clothing(
                name="Vintage Leather Rider Jacket",
                category="outerwear",
                color="#0f172a",
                image_path="/samples/2_garment_cloth.jpg",
                model_path="3d://mannequin/jacket-leather",
            ),
            Clothing(
                name="Relaxed Straight-Leg Denim Jeans",
                category="bottoms",
                color="#1d4ed8",
                image_path="/samples/1_target_person.jpg",
                model_path="3d://mannequin/denim-jeans",
            ),
            Clothing(
                name="Tapered Chino Trousers",
                category="bottoms",
                color="#b45309",
                image_path="/samples/1_target_person.jpg",
                model_path="3d://mannequin/chino-trousers",
            ),
        ]
        db.session.add_all(default_clothes)
        db.session.commit()


# =========================================================
# REST API ENDPOINTS (mini.py + Frontend Integration)
# =========================================================

# Home API / Web UI
@app.route("/", methods=["GET"])
def home():
    # If opened in a browser asking for HTML and combined_app.html exists, serve UI unless ?format=json
    accept = request.headers.get("Accept", "")
    if "text/html" in accept and request.args.get("format") != "json" and os.path.exists(COMBINED_HTML):
        return send_file(COMBINED_HTML)

    return jsonify({
        "message": "StyleMate Backend is running",
        "database": "SQLite",
        "status": "success"
    })


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


# Register a basic user profile
@app.route("/api/users", methods=["POST"])
def add_user():
    data = request.get_json(silent=True) or {}

    name = data.get("name")
    email = data.get("email")

    if not name or not email:
        return jsonify({"error": "Name and email are required"}), 400

    existing = User.query.filter_by(email=email).first()
    if existing:
        return jsonify({
            "message": "User already registered",
            "user_id": existing.id,
            "name": existing.name,
            "email": existing.email
        }), 200

    user = User(name=name, email=email)
    db.session.add(user)
    db.session.commit()

    return jsonify({
        "message": "User created",
        "user_id": user.id,
        "name": user.name,
        "email": user.email
    }), 201


@app.route("/api/users", methods=["GET"])
def list_users():
    users = User.query.all()
    return jsonify([
        {"id": u.id, "name": u.name, "email": u.email}
        for u in users
    ])


# Upload a full-body photo (Teammate's endpoint, now enhanced with AI auto-analysis)
@app.route("/api/upload", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return jsonify({"error": "Please attach an image"}), 400

    image = request.files["image"]
    user_id = request.form.get("user_id", "1")

    if not image.filename:
        return jsonify({"error": "No file selected"}), 400

    if not str(user_id).isdigit():
        return jsonify({"error": "Valid user_id is required"}), 400

    user_obj = db.session.get(User, int(user_id))
    if not user_obj:
        return jsonify({"error": "User not found"}), 404

    extension = os.path.splitext(image.filename)[1].lower()
    if not extension:
        extension = ".jpg"

    if extension not in [".jpg", ".jpeg", ".png"]:
        return jsonify({
            "error": "Only JPG, JPEG and PNG files are allowed"
        }), 400

    filename = secure_filename(image.filename) or "upload.jpg"
    unique_name = str(uuid4()) + "_" + filename
    save_path = os.path.join(UPLOAD_FOLDER, unique_name)

    image_bytes = image.read()
    with open(save_path, "wb") as f:
        f.write(image_bytes)

    ai_result = run_ai_fashion_analysis(image_bytes)

    analysis = Analysis(
        user_id=int(user_id),
        image_path=unique_name,
        body_measurements=json.dumps({
            **ai_result["body_measurements"],
            "recommended_clothes": ai_result["recommended_clothes"],
            "recommended_shoes": ai_result["recommended_shoes"],
            "recommended_sunglasses": ai_result["recommended_sunglasses"],
        }),
        face_shape=ai_result["face_shape"],
    )

    db.session.add(analysis)
    db.session.commit()

    return jsonify({
        "message": "Image uploaded and analyzed successfully",
        "analysis_id": analysis.id,
        "user_id": analysis.user_id,
        "image_path": unique_name,
        "status": "completed",
        **ai_result,
    }), 201


# Unified endpoint used by fashion-frontend (App.jsx) & combined_app.html
@app.route("/api/analyze", methods=["POST"])
def analyze_and_store():
    image_bytes = b""
    filename = "uploaded_person.jpg"
    user_id = 1

    if "user_id" in request.form and str(request.form["user_id"]).isdigit():
        user_id = int(request.form["user_id"])

    if "image" in request.files:
        img_file = request.files["image"]
        if img_file.filename:
            filename = secure_filename(img_file.filename)
        image_bytes = img_file.read()
    else:
        image_bytes = request.get_data()

    # Save to uploads/ and samples/
    unique_name = str(uuid4()) + "_" + (filename or "person.jpg")
    save_path = os.path.join(UPLOAD_FOLDER, unique_name)
    if image_bytes:
        with open(save_path, "wb") as f:
            f.write(image_bytes)
        sample_copy = os.path.join(SAMPLES_DIR, "partner_uploaded_person.jpg")
        with open(sample_copy, "wb") as f:
            f.write(image_bytes)

    ai_result = run_ai_fashion_analysis(image_bytes)

    # Persist into SQLite Analysis table
    if not db.session.get(User, user_id):
        user_id = 1

    analysis = Analysis(
        user_id=user_id,
        image_path=unique_name,
        body_measurements=json.dumps({
            **ai_result["body_measurements"],
            "recommended_clothes": ai_result["recommended_clothes"],
            "recommended_shoes": ai_result["recommended_shoes"],
            "recommended_sunglasses": ai_result["recommended_sunglasses"],
        }),
        face_shape=ai_result["face_shape"],
    )
    db.session.add(analysis)
    db.session.commit()

    return jsonify({
        **ai_result,
        "analysis_id": analysis.id,
        "user_id": user_id,
        "image_path": unique_name,
    })


# Store results produced by the AI/ML member
@app.route("/api/analysis", methods=["POST"])
def save_analysis():
    data = request.get_json(silent=True) or {}

    analysis_id = data.get("analysis_id")
    measurements = data.get("body_measurements")
    face_shape = data.get("face_shape")

    if analysis_id is None:
        return jsonify({"error": "analysis_id is required"}), 400

    analysis = db.session.get(Analysis, analysis_id)

    if analysis is None:
        return jsonify({"error": "Analysis record not found"}), 404

    if measurements is not None:
        analysis.body_measurements = json.dumps(measurements)

    if face_shape is not None:
        analysis.face_shape = face_shape

    db.session.commit()

    return jsonify({
        "message": "Analysis results saved",
        "analysis_id": analysis.id
    })


# Retrieve analysis results
@app.route("/api/analysis/<int:analysis_id>", methods=["GET"])
def get_analysis(analysis_id):
    analysis = db.session.get(Analysis, analysis_id)

    if analysis is None:
        return jsonify({"error": "Analysis not found"}), 404

    measurements = None

    if analysis.body_measurements:
        measurements = json.loads(analysis.body_measurements)

    return jsonify({
        "analysis_id": analysis.id,
        "user_id": analysis.user_id,
        "image_path": analysis.image_path,
        "body_measurements": measurements,
        "face_shape": analysis.face_shape
    })


# Add clothing catalog items
@app.route("/api/clothing", methods=["POST"])
def add_clothing():
    data = request.get_json(silent=True) or {}

    name = data.get("name")
    category = data.get("category")

    if not name or not category:
        return jsonify({
            "error": "Name and category are required"
        }), 400

    item = Clothing(
        name=name,
        category=category,
        color=data.get("color"),
        image_path=data.get("image_path"),
        model_path=data.get("model_path")
    )

    db.session.add(item)
    db.session.commit()

    return jsonify({
        "message": "Clothing item added",
        "item_id": item.id
    }), 201


# View clothing catalog
@app.route("/api/clothing", methods=["GET"])
def get_clothing():
    items = Clothing.query.all()

    return jsonify([
        {
            "id": item.id,
            "name": item.name,
            "category": item.category,
            "color": item.color,
            "image_path": item.image_path,
            "model_path": item.model_path
        }
        for item in items
    ])


# Save an outfit selected by the user
@app.route("/api/outfits/save", methods=["POST"])
def save_outfit():
    data = request.get_json(silent=True) or {}

    user_id = data.get("user_id", 1)
    clothing_id = data.get("clothing_id")
    clothing_name = data.get("clothing_name")
    accessory_id = data.get("accessory_id")

    # Allow saving by clothing_name if frontend passes names instead of numeric ID
    if clothing_id is None and clothing_name:
        matched = Clothing.query.filter(
            Clothing.name.ilike(f"%{clothing_name}%")
        ).first()
        if not matched:
            matched = Clothing(
                name=clothing_name,
                category="custom",
                color="#4f46e5",
                image_path="/samples/4_vton_synthesis_result.jpg",
                model_path="3d://mannequin/custom",
            )
            db.session.add(matched)
            db.session.commit()
        clothing_id = matched.id

    if user_id is None or clothing_id is None:
        return jsonify({
            "error": "user_id and clothing_id are required"
        }), 400

    if not db.session.get(User, int(user_id)):
        return jsonify({"error": "User not found"}), 404

    clothing_obj = db.session.get(Clothing, int(clothing_id))
    if not clothing_obj:
        return jsonify({"error": "Clothing item not found"}), 404

    outfit = SavedOutfit(
        user_id=int(user_id),
        clothing_id=int(clothing_id),
        accessory_id=accessory_id
    )

    db.session.add(outfit)
    db.session.commit()

    return jsonify({
        "message": "Outfit saved successfully",
        "outfit_id": outfit.id,
        "clothing_name": clothing_obj.name,
        "clothing_id": clothing_obj.id,
        "accessory_id": outfit.accessory_id
    }), 201


# View a user's saved outfits
@app.route("/api/outfits/<int:user_id>", methods=["GET"])
def get_outfits(user_id):
    outfits = SavedOutfit.query.filter_by(user_id=user_id).all()
    results = []
    for outfit in outfits:
        clothing_obj = db.session.get(Clothing, outfit.clothing_id)
        results.append({
            "outfit_id": outfit.id,
            "user_id": outfit.user_id,
            "clothing_id": outfit.clothing_id,
            "clothing_name": clothing_obj.name if clothing_obj else f"Item #{outfit.clothing_id}",
            "clothing_color": clothing_obj.color if clothing_obj else "#4f46e5",
            "accessory_id": outfit.accessory_id
        })
    return jsonify(results)


# Provide selected item IDs to the 3D frontend module
@app.route("/api/try-on", methods=["POST"])
def try_on():
    data = request.get_json(silent=True) or {}

    clothing_id = data.get("clothing_id")
    accessory_id = data.get("accessory_id")

    if clothing_id is None:
        return jsonify({"error": "clothing_id is required"}), 400

    item = db.session.get(Clothing, int(clothing_id))

    if item is None:
        return jsonify({"error": "Clothing item not found"}), 404

    return jsonify({
        "message": "Items ready for 3D visualization",
        "clothing_id": item.id,
        "clothing_name": item.name,
        "clothing_color": item.color,
        "clothing_model": item.model_path,
        "texture_url": "/samples/4_vton_synthesis_result.jpg",
        "accessory_id": accessory_id
    })


# Serve sample images and uploaded images
@app.route("/samples/<path:filename>", methods=["GET"])
def serve_samples(filename):
    return send_from_directory(SAMPLES_DIR, filename)


@app.route("/uploads/<path:filename>", methods=["GET"])
def serve_uploads(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


# Serve web_app/ static files (3D studio, GLB model, JS loaders)
@app.route("/web_app/<path:filename>", methods=["GET"])
def serve_web_app(filename):
    return send_from_directory(WEB_APP_DIR, filename)


# Alias: /models/ → web_app/models/ (backward-compatible)
@app.route("/models/<path:filename>", methods=["GET"])
def serve_models(filename):
    models_dir = os.path.join(WEB_APP_DIR, "models")
    return send_from_directory(models_dir, filename)


# Serve web_app/index.html at /studio
@app.route("/studio", methods=["GET"])
def studio():
    return send_file(os.path.join(WEB_APP_DIR, "index.html"))


# Create database tables and seed initial catalog
with app.app_context():
    db.create_all()
    seed_default_data()


if __name__ == "__main__":
    print("=" * 68)
    print("  STYLEMATE FLASK + SQLITE + AI 3D BACKEND (PORT 5000)")
    print("=" * 68)
    print(f" [OK] SQLite Database: {DB_PATH}")
    print(f" [OK] Uploads Folder:  {UPLOAD_FOLDER}")
    print(" [OK] Server URL:      http://127.0.0.1:5000")
    print(" [OK] 3D Studio:       http://127.0.0.1:5000/studio")
    print("=" * 68)

    def open_browser():
        time.sleep(1.5)
        webbrowser.open("http://127.0.0.1:5000")

    threading.Thread(target=open_browser, daemon=True).start()
    app.run(debug=False, port=5000)