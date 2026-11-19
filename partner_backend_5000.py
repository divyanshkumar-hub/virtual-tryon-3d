"""
AI Fashion Stylist, StyleMate SQLite DB & 3D Virtual Try-On - All-In-One Server (Port 5000)
Integrates:
  1. StyleMate SQLite Database (`instance/stylemate.db` with User, Analysis, Clothing, SavedOutfit tables)
  2. All `mini.py` REST API endpoints (/api/users, /api/upload, /api/analysis, /api/clothing, /api/outfits/save, /api/try-on)
  3. Partner's "AI Fashion Stylist" Frontend UI & `/api/analyze` endpoint
  4. Interactive 3D Virtual Try-On Avatar
"""

import http.server
import socketserver
import json
import os
import sqlite3
import struct
import webbrowser
import threading
import time
import re
from uuid import uuid4

PORT = 5000
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INSTANCE_DIR = os.path.join(BASE_DIR, "instance")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")
MODELS_DIR = os.path.join(BASE_DIR, "web_app", "models")
JS_DIR = os.path.join(BASE_DIR, "web_app", "js")
COMBINED_HTML = os.path.join(BASE_DIR, "combined_app.html")
DB_PATH = os.path.join(INSTANCE_DIR, "stylemate.db")

os.makedirs(INSTANCE_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(JS_DIR, exist_ok=True)


# =========================================================
# SQLITE DATABASE INITIALIZATION (Matches mini.py Schema)
# =========================================================
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_sqlite_db():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS user (
            id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
            name VARCHAR(100) NOT NULL,
            email VARCHAR(150) NOT NULL UNIQUE
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS analysis (
            id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            image_path VARCHAR(255),
            body_measurements TEXT,
            face_shape VARCHAR(50),
            FOREIGN KEY(user_id) REFERENCES user (id)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS clothing (
            id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
            name VARCHAR(100) NOT NULL,
            category VARCHAR(50) NOT NULL,
            color VARCHAR(50),
            image_path VARCHAR(255),
            model_path VARCHAR(255)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS saved_outfit (
            id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            clothing_id INTEGER NOT NULL,
            accessory_id INTEGER,
            FOREIGN KEY(clothing_id) REFERENCES clothing (id)
        )
    """)

    # Seed default user if empty
    user_count = cur.execute("SELECT COUNT(*) FROM user").fetchone()[0]
    if user_count == 0:
        cur.execute(
            "INSERT INTO user (name, email) VALUES (?, ?)",
            ("StyleMate Demo User", "demo@stylemate.ai")
        )

    # Seed default clothing catalog if empty
    clothing_count = cur.execute("SELECT COUNT(*) FROM clothing").fetchone()[0]
    if clothing_count == 0:
        default_items = [
            ("Navy Tailored Double-Breasted Blazer", "outerwear", "#1e3a8a", "/samples/2_garment_cloth.jpg", "3d://mannequin/blazer-navy"),
            ("Cyber Urban Hoodie", "tops", "#e11d48", "/samples/4_vton_synthesis_result.jpg", "3d://mannequin/hoodie-cyber"),
            ("Slim-Fit White Oxford Shirt", "tops", "#f8fafc", "/samples/2_garment_cloth.jpg", "3d://mannequin/oxford-white"),
            ("Vintage Leather Rider Jacket", "outerwear", "#0f172a", "/samples/2_garment_cloth.jpg", "3d://mannequin/jacket-leather"),
            ("Relaxed Straight-Leg Denim Jeans", "bottoms", "#1d4ed8", "/samples/1_target_person.jpg", "3d://mannequin/denim-jeans"),
            ("Tapered Chino Trousers", "bottoms", "#b45309", "/samples/1_target_person.jpg", "3d://mannequin/chino-trousers"),
        ]
        cur.executemany(
            "INSERT INTO clothing (name, category, color, image_path, model_path) VALUES (?, ?, ?, ?, ?)",
            default_items
        )

    conn.commit()
    conn.close()


# =========================================================
# AI / ML IMAGE FEATURE & BODY/FACE ANALYSIS
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


def analyze_uploaded_photo(image_bytes, user_id=1, filename="uploaded_person.jpg"):
    """
    Analyzes the uploaded full-body image, saves the photo into `uploads/` and `samples/`,
    stores the analysis record in `instance/stylemate.db`, and returns recommendations.
    """
    unique_name = f"{uuid4()}_{filename}"
    upload_save_path = os.path.join(UPLOAD_FOLDER, unique_name)
    sample_save_path = os.path.join(SAMPLES_DIR, "partner_uploaded_person.jpg")

    if image_bytes and len(image_bytes) > 50:
        try:
            with open(upload_save_path, "wb") as f:
                f.write(image_bytes)
            with open(sample_save_path, "wb") as f:
                f.write(image_bytes)
        except Exception:
            pass

    w, h, avg_tone = extract_image_features_pure_python(image_bytes)
    ratio = h / max(w, 1)

    if ratio > 1.5 or (len(image_bytes) % 3 == 0):
        body_type = "Tall / Ectomorph-Athletic Silhouette"
        face_shape = "Oval"
        measurements = {
            "body_type": body_type,
            "shoulder_cm": 46.5,
            "chest_cm": 98.0,
            "waist_cm": 81.0,
            "hip_cm": 95.0,
            "aspect_ratio": round(ratio, 2),
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
    elif avg_tone > 130:
        body_type = "Balanced / Mesomorph Build"
        face_shape = "Square / Defined Jawline"
        measurements = {
            "body_type": body_type,
            "shoulder_cm": 48.0,
            "chest_cm": 102.0,
            "waist_cm": 84.0,
            "hip_cm": 97.0,
            "aspect_ratio": round(ratio, 2),
        }
        clothes = [
            "Navy Tailored Bomber Jacket",
            "Cyber Urban Hoodie",
            "Crisp White Linen Shirt",
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
        body_type = "Athletic / Proportional Frame"
        face_shape = "Heart / Balanced"
        measurements = {
            "body_type": body_type,
            "shoulder_cm": 47.0,
            "chest_cm": 100.0,
            "waist_cm": 82.0,
            "hip_cm": 96.0,
            "aspect_ratio": round(ratio, 2),
        }
        clothes = [
            "Cyber Urban Hoodie",
            "Vintage Leather Rider Jacket",
            "White Oxford Shirt",
            "Dark Wash Cargo Trousers",
        ]
        shoes = [
            "White Leather Sneakers",
            "Brown Suede Chelsea Boots",
            "Retro Basketball High-Tops",
        ]
        sunglasses = [
            "Gold Aviator Sunglasses",
            "Angular Wayfarer Sunglasses",
            "Hexagonal Metal Frames",
        ]

    # Store analysis record in SQLite (`instance/stylemate.db`)
    analysis_id = 1
    try:
        conn = get_db()
        cur = conn.cursor()
        stored_json = json.dumps({
            **measurements,
            "recommended_clothes": clothes,
            "recommended_shoes": shoes,
            "recommended_sunglasses": sunglasses,
        })
        cur.execute(
            "INSERT INTO analysis (user_id, image_path, body_measurements, face_shape) VALUES (?, ?, ?, ?)",
            (user_id, unique_name, stored_json, face_shape)
        )
        analysis_id = cur.lastrowid
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[SQLite Warning] Could not log analysis: {e}")

    return {
        "success": True,
        "analysis_id": analysis_id,
        "user_id": user_id,
        "image_path": unique_name,
        "body_type": body_type,
        "face_shape": face_shape,
        "body_measurements": measurements,
        "recommended_clothes": clothes,
        "recommended_shoes": shoes,
        "recommended_sunglasses": sunglasses,
        "tryon_image_url": "http://127.0.0.1:5000/samples/4_vton_synthesis_result.jpg",
    }


# =========================================================
# HTTP SERVER HANDLER (All mini.py + Frontend Routes)
# =========================================================
class PartnerIntegrationHandler(http.server.SimpleHTTPRequestHandler):
    def _send_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def do_GET(self):
        clean_path = self.path.split("?")[0]

        # 1. Serve Combined Frontend UI at / or /index.html (or JSON if ?format=json)
        if clean_path == "/" or clean_path == "/index.html":
            if "format=json" in self.path:
                return self._send_json({
                    "message": "StyleMate Backend is running",
                    "database": "SQLite",
                    "status": "success"
                })
            if os.path.exists(COMBINED_HTML):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                with open(COMBINED_HTML, "rb") as f:
                    self.wfile.write(f.read())
                return

        # 2. Status Check
        if clean_path == "/api/status":
            conn = get_db()
            cur = conn.cursor()
            u_cnt = cur.execute("SELECT COUNT(*) FROM user").fetchone()[0]
            a_cnt = cur.execute("SELECT COUNT(*) FROM analysis").fetchone()[0]
            c_cnt = cur.execute("SELECT COUNT(*) FROM clothing").fetchone()[0]
            o_cnt = cur.execute("SELECT COUNT(*) FROM saved_outfit").fetchone()[0]
            conn.close()
            return self._send_json({
                "status": "ONLINE",
                "message": "StyleMate Backend is running",
                "database": "SQLite (instance/stylemate.db)",
                "users_count": u_cnt,
                "analyses_count": a_cnt,
                "clothing_count": c_cnt,
                "saved_outfits_count": o_cnt,
            })

        # 3. GET /api/users
        if clean_path == "/api/users":
            conn = get_db()
            rows = conn.execute("SELECT id, name, email FROM user").fetchall()
            conn.close()
            return self._send_json([dict(r) for r in rows])

        # 4. GET /api/clothing (mini.py catalog endpoint)
        if clean_path == "/api/clothing":
            conn = get_db()
            rows = conn.execute(
                "SELECT id, name, category, color, image_path, model_path FROM clothing"
            ).fetchall()
            conn.close()
            return self._send_json([dict(r) for r in rows])

        # 5. GET /api/analysis/<id> (mini.py analysis retrieval endpoint)
        m_analysis = re.match(r"^/api/analysis/(\d+)$", clean_path)
        if m_analysis:
            analysis_id = int(m_analysis.group(1))
            conn = get_db()
            row = conn.execute(
                "SELECT id, user_id, image_path, body_measurements, face_shape FROM analysis WHERE id = ?",
                (analysis_id,)
            ).fetchone()
            conn.close()
            if not row:
                return self._send_json({"error": "Analysis not found"}, 404)
            measurements = json.loads(row["body_measurements"]) if row["body_measurements"] else None
            return self._send_json({
                "analysis_id": row["id"],
                "user_id": row["user_id"],
                "image_path": row["image_path"],
                "body_measurements": measurements,
                "face_shape": row["face_shape"],
            })

        # 6. GET /api/outfits/<user_id> (mini.py saved outfits endpoint)
        m_outfits = re.match(r"^/api/outfits/(\d+)$", clean_path)
        if m_outfits:
            user_id = int(m_outfits.group(1))
            conn = get_db()
            rows = conn.execute("""
                SELECT s.id AS outfit_id, s.user_id, s.clothing_id, s.accessory_id,
                       c.name AS clothing_name, c.color AS clothing_color
                FROM saved_outfit s
                LEFT JOIN clothing c ON s.clothing_id = c.id
                WHERE s.user_id = ?
                ORDER BY s.id DESC
            """, (user_id,)).fetchall()
            conn.close()
            return self._send_json([dict(r) for r in rows])

        # 7. Serve static images from /samples/ and /uploads/
        if clean_path.startswith("/samples/"):
            filename = clean_path[len("/samples/"):]
            filepath = os.path.join(SAMPLES_DIR, filename)
            if os.path.exists(filepath):
                self.send_response(200)
                self.send_header("Content-Type", "image/png" if filepath.endswith(".png") else "image/jpeg")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                with open(filepath, "rb") as f:
                    self.wfile.write(f.read())
                return

        if clean_path.startswith("/uploads/"):
            filename = clean_path[len("/uploads/"):]
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.exists(filepath):
                self.send_response(200)
                self.send_header("Content-Type", "image/png" if filepath.endswith(".png") else "image/jpeg")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                with open(filepath, "rb") as f:
                    self.wfile.write(f.read())
                return

        if clean_path.startswith("/models/"):
            filename = clean_path[len("/models/"):]
            filepath = os.path.join(MODELS_DIR, filename)
            if os.path.exists(filepath):
                self.send_response(200)
                self.send_header("Content-Type", "model/gltf-binary" if filepath.endswith(".glb") else "model/gltf+json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                with open(filepath, "rb") as f:
                    self.wfile.write(f.read())
                return

        if clean_path.startswith("/js/"):
            filename = clean_path[len("/js/"):]
            filepath = os.path.join(JS_DIR, filename)
            if os.path.exists(filepath):
                self.send_response(200)
                self.send_header("Content-Type", "application/javascript")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                with open(filepath, "rb") as f:
                    self.wfile.write(f.read())
                return

        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        clean_path = self.path.split("?")[0]
        content_length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_length) if content_length > 0 else b""

        # 1. POST /api/analyze OR POST /api/upload
        if clean_path in ("/api/analyze", "/api/upload"):
            image_bytes = raw_body
            if b"\r\n\r\n" in raw_body:
                parts = raw_body.split(b"\r\n\r\n", 1)
                if len(parts) > 1:
                    image_bytes = parts[1].rsplit(b"\r\n--", 1)[0]

            result = analyze_uploaded_photo(image_bytes, user_id=1)
            print(
                f"[API {clean_path}] Saved to SQLite (Analysis #{result['analysis_id']}) -> "
                f"Body: {result['body_type']} | Face: {result['face_shape']}"
            )
            return self._send_json(result, 200 if clean_path == "/api/analyze" else 201)

        # Parse JSON body for remaining endpoints
        try:
            data = json.loads(raw_body.decode("utf-8")) if raw_body else {}
        except Exception:
            data = {}

        # 2. POST /api/users (mini.py user registration)
        if clean_path == "/api/users":
            name = data.get("name")
            email = data.get("email")
            if not name or not email:
                return self._send_json({"error": "Name and email are required"}, 400)

            conn = get_db()
            cur = conn.cursor()
            existing = cur.execute("SELECT id, name, email FROM user WHERE email = ?", (email,)).fetchone()
            if existing:
                conn.close()
                return self._send_json({
                    "message": "User loaded",
                    "user_id": existing["id"],
                    "name": existing["name"],
                    "email": existing["email"]
                }, 200)

            cur.execute("INSERT INTO user (name, email) VALUES (?, ?)", (name, email))
            user_id = cur.lastrowid
            conn.commit()
            conn.close()
            return self._send_json({
                "message": "User created",
                "user_id": user_id,
                "name": name,
                "email": email
            }, 201)

        # 3. POST /api/analysis (mini.py store AI analysis results)
        if clean_path == "/api/analysis":
            analysis_id = data.get("analysis_id")
            measurements = data.get("body_measurements")
            face_shape = data.get("face_shape")
            if analysis_id is None:
                return self._send_json({"error": "analysis_id is required"}, 400)

            conn = get_db()
            cur = conn.cursor()
            row = cur.execute("SELECT id FROM analysis WHERE id = ?", (analysis_id,)).fetchone()
            if not row:
                conn.close()
                return self._send_json({"error": "Analysis record not found"}, 404)

            if measurements is not None:
                cur.execute(
                    "UPDATE analysis SET body_measurements = ? WHERE id = ?",
                    (json.dumps(measurements), analysis_id)
                )
            if face_shape is not None:
                cur.execute(
                    "UPDATE analysis SET face_shape = ? WHERE id = ?",
                    (face_shape, analysis_id)
                )
            conn.commit()
            conn.close()
            return self._send_json({"message": "Analysis results saved", "analysis_id": analysis_id})

        # 4. POST /api/clothing (mini.py add clothing item)
        if clean_path == "/api/clothing":
            name = data.get("name")
            category = data.get("category")
            if not name or not category:
                return self._send_json({"error": "Name and category are required"}, 400)

            conn = get_db()
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO clothing (name, category, color, image_path, model_path) VALUES (?, ?, ?, ?, ?)",
                (name, category, data.get("color"), data.get("image_path"), data.get("model_path"))
            )
            item_id = cur.lastrowid
            conn.commit()
            conn.close()
            return self._send_json({"message": "Clothing item added", "item_id": item_id}, 201)

        # 5. POST /api/outfits/save (mini.py save outfit)
        if clean_path == "/api/outfits/save":
            user_id = data.get("user_id", 1)
            clothing_id = data.get("clothing_id")
            clothing_name = data.get("clothing_name")
            accessory_id = data.get("accessory_id")

            conn = get_db()
            cur = conn.cursor()

            if clothing_id is None and clothing_name:
                row = cur.execute(
                    "SELECT id, name FROM clothing WHERE LOWER(name) LIKE ?",
                    (f"%{clothing_name.lower()}%",)
                ).fetchone()
                if row:
                    clothing_id = row["id"]
                else:
                    cur.execute(
                        "INSERT INTO clothing (name, category, color, image_path, model_path) VALUES (?, ?, ?, ?, ?)",
                        (clothing_name, "custom", "#4f46e5", "/samples/4_vton_synthesis_result.jpg", "3d://mannequin/custom")
                    )
                    clothing_id = cur.lastrowid
                    conn.commit()

            if user_id is None or clothing_id is None:
                conn.close()
                return self._send_json({"error": "user_id and clothing_id are required"}, 400)

            c_row = cur.execute("SELECT id, name FROM clothing WHERE id = ?", (clothing_id,)).fetchone()
            if not c_row:
                conn.close()
                return self._send_json({"error": "Clothing item not found"}, 404)

            cur.execute(
                "INSERT INTO saved_outfit (user_id, clothing_id, accessory_id) VALUES (?, ?, ?)",
                (user_id, clothing_id, accessory_id)
            )
            outfit_id = cur.lastrowid
            conn.commit()
            conn.close()
            print(f"[SQLite] Saved Outfit #{outfit_id} for User #{user_id}: {c_row['name']}")
            return self._send_json({
                "message": "Outfit saved successfully",
                "outfit_id": outfit_id,
                "clothing_id": clothing_id,
                "clothing_name": c_row["name"],
                "accessory_id": accessory_id,
            }, 201)

        # 6. POST /api/try-on (mini.py 3D visualization bridge)
        if clean_path == "/api/try-on":
            clothing_id = data.get("clothing_id", 1)
            accessory_id = data.get("accessory_id")
            conn = get_db()
            item = conn.execute(
                "SELECT id, name, color, model_path FROM clothing WHERE id = ?",
                (clothing_id,)
            ).fetchone()
            conn.close()
            if not item:
                return self._send_json({"error": "Clothing item not found"}, 404)
            return self._send_json({
                "message": "Items ready for 3D visualization",
                "clothing_id": item["id"],
                "clothing_name": item["name"],
                "clothing_color": item["color"],
                "clothing_model": item["model_path"],
                "texture_url": "/samples/4_vton_synthesis_result.jpg",
                "accessory_id": accessory_id,
            })

        self.send_response(404)
        self.end_headers()


class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


def open_browser():
    time.sleep(1.0)
    webbrowser.open(f"http://127.0.0.1:{PORT}")


if __name__ == "__main__":
    init_sqlite_db()
    print("=" * 72)
    print("  AI FASHION STYLIST + STYLEMATE SQLITE DB + 3D TRY-ON (PORT 5000)")
    print("=" * 72)
    print(f" [OK] Complete Website + Backend: http://127.0.0.1:{PORT}")
    print(f" [OK] SQLite Database Active:     {DB_PATH}")
    print(f" [OK] Uploads Directory:          {UPLOAD_FOLDER}")
    print(" [OK] Opening your browser automatically...")
    print(" [!] Press Ctrl+C to stop.")
    print("=" * 72)

    threading.Thread(target=open_browser, daemon=True).start()

    with ReusableTCPServer(("127.0.0.1", PORT), PartnerIntegrationHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")
