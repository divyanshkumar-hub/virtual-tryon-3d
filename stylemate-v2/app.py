"""
StyleMate v2 - Minimalist Flask Backend & Virtual Try-On API
"""

import os
import json
from uuid import uuid4
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from vision_analyzer import analyze_person_image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INSTANCE_DIR = os.path.join(BASE_DIR, "instance")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")
WEB_APP_DIR = os.path.join(BASE_DIR, "web_app")
STATIC_HTML = os.path.join(BASE_DIR, "index.html")
REACT_DIST_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "fashion-frontend", "fashion-frontend", "dist"))

for path in (INSTANCE_DIR, UPLOAD_DIR):
    os.makedirs(path, exist_ok=True)

app = Flask(__name__, instance_path=INSTANCE_DIR)
CORS(app)
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{os.path.join(INSTANCE_DIR, 'stylemate.db')}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
db = SQLAlchemy(app)


# ── Database Models ──────────────────────────────────────────
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)


class Clothing(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    category = db.Column(db.String(50), nullable=False)  # tops, bottoms, outerwear
    color = db.Column(db.String(30), default="#4f46e5")
    image_url = db.Column(db.String(255), default="/samples/2_garment_cloth.jpg")
    model_tag = db.Column(db.String(50), default="top")


class SavedOutfit(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    clothing_id = db.Column(db.Integer, nullable=False)
    clothing_name = db.Column(db.String(120), nullable=False)
    clothing_color = db.Column(db.String(30), default="#4f46e5")
    accessory = db.Column(db.String(100), nullable=True)


# ── API Endpoints ────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    if os.path.exists(STATIC_HTML):
        return send_file(STATIC_HTML)
    return jsonify({"message": "StyleMate v2 API is running", "status": "online"})


@app.route("/favicon.ico", methods=["GET"])
def favicon():
    return "", 204


@app.route("/api/status", methods=["GET"])
def api_status():
    return jsonify({
        "status": "online",
        "service": "StyleMate v2 Virtual Try-On Studio",
        "users_count": User.query.count(),
        "clothing_count": Clothing.query.count(),
        "saved_outfits_count": SavedOutfit.query.count(),
    })


@app.route("/api/upload", methods=["POST"])
@app.route("/api/analyze", methods=["POST"])
def upload_and_analyze():
    user_id = request.form.get("user_id", 1)
    height_cm = float(request.form.get("height_cm", 175.0))

    image_bytes = None
    filename = "photo.jpg"

    if "image" in request.files:
        file = request.files["image"]
        if file.filename:
            filename = secure_filename(file.filename)
            image_bytes = file.read()
    if not image_bytes:
        image_bytes = request.get_data()

    if not image_bytes:
        sample_path = os.path.join(SAMPLES_DIR, "1_target_person.jpg")
        if os.path.exists(sample_path):
            with open(sample_path, "rb") as f:
                image_bytes = f.read()
        else:
            return jsonify({"error": "No image data provided"}), 400

    unique_name = f"{uuid4()}_{filename}"
    save_path = os.path.join(UPLOAD_DIR, unique_name)
    with open(save_path, "wb") as f:
        f.write(image_bytes)

    result = analyze_person_image(image_bytes, user_height_cm=height_cm)
    result["image_path"] = f"/uploads/{unique_name}"
    result["user_id"] = user_id
    return jsonify(result), 200


@app.route("/api/clothing", methods=["GET"])
def get_clothing():
    items = Clothing.query.all()
    return jsonify([{
        "id": c.id, "name": c.name, "category": c.category,
        "color": c.color, "image_url": c.image_url, "model_tag": c.model_tag
    } for c in items])


@app.route("/api/clothing", methods=["POST"])
def add_clothing():
    data = request.get_json() or {}
    name = data.get("name")
    category = data.get("category", "tops")
    if not name:
        return jsonify({"error": "name is required"}), 400
    item = Clothing(
        name=name, category=category,
        color=data.get("color", "#4f46e5"),
        image_url=data.get("image_url", "/samples/2_garment_cloth.jpg"),
        model_tag=data.get("model_tag", "top")
    )
    db.session.add(item)
    db.session.commit()
    return jsonify({"message": "Item added", "id": item.id}), 201


@app.route("/api/try-on", methods=["POST"])
def try_on():
    data = request.get_json() or {}
    clothing_id = data.get("clothing_id")
    item = db.session.get(Clothing, clothing_id) if clothing_id else None
    if not item:
        item = Clothing.query.first()
    return jsonify({
        "message": "3D visualization parameters ready",
        "clothing_id": item.id if item else 1,
        "clothing_name": item.name if item else "Cyber Urban Hoodie",
        "color": item.color if item else "#e11d48",
        "category": item.category if item else "tops",
        "texture_url": "/samples/4_vton_synthesis_result.jpg"
    })


@app.route("/api/users", methods=["GET"])
def get_users():
    users = User.query.all()
    return jsonify([{"id": u.id, "name": u.name, "email": u.email} for u in users])


@app.route("/api/users", methods=["POST"])
def register_user():
    data = request.get_json() or {}
    name = data.get("name")
    email = data.get("email")
    if not name or not email:
        return jsonify({"error": "name and email required"}), 400
    user = User.query.filter_by(email=email).first()
    if not user:
        user = User(name=name, email=email)
        db.session.add(user)
        db.session.commit()
    return jsonify({"message": "User registered", "user_id": user.id, "name": user.name}), 201


@app.route("/api/outfits", methods=["POST"])
@app.route("/api/outfits/save", methods=["POST"])
def save_outfit():
    data = request.get_json() or {}
    user_id = int(data.get("user_id", 1))
    clothing_id = int(data.get("clothing_id", 1))
    clothing_name = data.get("clothing_name", "Selected Garment")
    clothing_color = data.get("clothing_color", "#4f46e5")
    accessory = data.get("accessory") or (data.get("accessory_id") and "Gold Aviators")

    outfit = SavedOutfit(
        user_id=user_id,
        clothing_id=clothing_id,
        clothing_name=clothing_name,
        clothing_color=clothing_color,
        accessory=accessory
    )
    db.session.add(outfit)
    db.session.commit()
    return jsonify({
        "message": "Outfit saved",
        "outfit_id": outfit.id,
        "clothing_name": clothing_name,
        "clothing_id": clothing_id
    }), 201


@app.route("/api/outfits/<int:user_id>", methods=["GET"])
def get_user_outfits(user_id):
    outfits = SavedOutfit.query.filter_by(user_id=user_id).all()
    return jsonify([{
        "outfit_id": o.id, "user_id": o.user_id, "clothing_id": o.clothing_id,
        "clothing_name": o.clothing_name, "clothing_color": o.clothing_color,
        "accessory": o.accessory
    } for o in outfits])


# Static assets
@app.route("/models/<path:filename>")
def serve_models(filename):
    models_dir = os.path.join(WEB_APP_DIR, "models")
    return send_from_directory(models_dir, filename)


@app.route("/uploads/<path:filename>")
def serve_uploads(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/samples/<path:filename>")
def serve_samples(filename):
    return send_from_directory(SAMPLES_DIR, filename)


@app.route("/web_app/<path:filename>")
def serve_web_app(filename):
    return send_from_directory(WEB_APP_DIR, filename)


# Serve partner's built React frontend directly from Flask
@app.route("/react")
def serve_react():
    if os.path.exists(REACT_DIST_DIR):
        return send_from_directory(REACT_DIST_DIR, "index.html")
    return "React build not found", 404


@app.route("/assets/<path:filename>")
def serve_react_assets(filename):
    if os.path.exists(REACT_DIST_DIR):
        return send_from_directory(os.path.join(REACT_DIST_DIR, "assets"), filename)
    return "Not found", 404


def seed_database():
    with app.app_context():
        db.create_all()
        if not User.query.first():
            db.session.add(User(name="Demo Stylist", email="demo@stylemate.ai"))
            db.session.commit()
        if not Clothing.query.first():
            defaults = [
                Clothing(name="Navy Tailored Double-Breasted Blazer", category="outerwear", color="#1e3a8a", model_tag="blazer"),
                Clothing(name="Cyber Urban Street Hoodie", category="tops", color="#e11d48", model_tag="hoodie"),
                Clothing(name="Classic White Oxford Shirt", category="tops", color="#f8fafc", model_tag="oxford"),
                Clothing(name="Vintage Leather Rider Jacket", category="outerwear", color="#0f172a", model_tag="leather"),
                Clothing(name="Athletic Performance Fitted Tee", category="tops", color="#059669", model_tag="tshirt"),
                Clothing(name="Tapered Tan Chino Trousers", category="bottoms", color="#b45309", model_tag="chinos"),
                Clothing(name="Selvedge Raw Dark Denim Jeans", category="bottoms", color="#1d4ed8", model_tag="denim"),
            ]
            db.session.add_all(defaults)
            db.session.commit()


seed_database()

if __name__ == "__main__":
    print("=" * 64)
    print("  StyleMate v2 - Clean REST API & 3D Virtual Try-On")
    print("  http://127.0.0.1:5000")
    print("=" * 64)
    app.run(debug=False, port=5000)
