# 👗 Virtual Try-On 3D Studio

> **AI-powered virtual fashion try-on** with a real-time 3D mannequin, body analysis, and outfit recommendations.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📸 **Real AI Body Measurements** | MediaPipe PoseLandmarker (33 anatomical keypoints) + OpenCV extracts real shoulder, chest, waist, hip, torso, arm & leg dimensions in cm |
| 📏 **Personalized Sizing** | Dynamically recommends exact clothing sizes (XS–XXL) based on computed chest and shoulder width |
| 👕 **Clothing Catalog** | Browse garments by category (tops, bottoms, dresses, outerwear, accessories) |
| 🧍 **Dynamic 3D Avatar Try-On** | Three.js mannequin auto-scales to match user's actual body measurements + real-time 360° orbit controls |
| 💾 **Save Outfits** | Save & retrieve outfit combinations tied to your user account in SQLite DB |
| 🎨 **Quick Style Chips** | One-click outfit presets for Casual, Formal, Party, etc. |
| 🤖 **AI Recommendations** | Intelligent matching based on body shape (Inverted Triangle, Hourglass, Pear, Rectangle) + color analysis |

---

## 🚀 Quick Start

### 1. Install Python dependencies
```powershell
py -m pip install -r requirements.txt
```

### 2. Run the Flask backend
```powershell
py mini.py
```
> The server starts on **http://127.0.0.1:5000** and auto-seeds the database on first run.

### 3. Open the app
Navigate to **http://127.0.0.1:5000** in your browser.

---

## 🗂️ Project Structure

```
virtual_tryon_3d/
│
├── mini.py                          # 🔥 Main Flask backend (port 5000)
├── partner_backend_5000.py          # ⚡ Zero-dependency fallback server
├── server.py                        # Static file server (port 8000)
├── run_pipeline_local.py            # Standalone AI pipeline (numpy + PIL)
│
├── combined_app.html                # 🌐 Full integrated frontend (single file)
├── requirements.txt                 # Python dependencies
│
├── run_mini_flask.bat               # Launch mini.py
├── run_partner_backend.bat          # Launch zero-dep server
├── run_project.bat                  # Launch static server
│
├── Phase1_AI_Model_Pipeline.ipynb   # Google Colab notebook (Phase 1 AI pipeline)
│
├── README.md
├── PROJECT_REPORT_PHASE1.md
├── PRESENTATION_SCRIPT.md
├── INTEGRATION_GUIDE.md
│
└── web_app/
    ├── index.html                   # 3D Studio standalone page
    ├── js/GLTFLoader.js             # Three.js GLTF loader
    └── models/realistic_human.glb  # 3D human avatar model (7.8 MB)
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serves combined_app.html |
| `GET` | `/api/status` | Health check |
| `POST` | `/api/upload` | Upload photo + run AI analysis |
| `POST` | `/api/analyze` | Analyze body type + get recommendations |
| `GET` | `/api/clothing` | Get clothing catalog |
| `POST` | `/api/outfits/save` | Save an outfit |
| `GET` | `/api/outfits/<user_id>` | Get saved outfits |
| `POST` | `/api/try-on` | Trigger 3D try-on data |

---

## 🛠️ Tech Stack

- **Backend:** Python 3.x, Flask, Flask-SQLAlchemy, Flask-CORS
- **Database:** SQLite (`instance/stylemate.db`)
- **Frontend:** HTML5, CSS3, JavaScript (ES6)
- **3D Engine:** Three.js r128 (CDN), GLTFLoader, OrbitControls
- **AI Pipeline:** NumPy, Pillow (image synthesis + SSIM/PSNR metrics)
- **Colab Notebook:** Phase 1 AI pipeline demo

---

## 👥 Team

| Member | Role |
|--------|------|
| **Backend Dev** | Flask API, SQLite schema, AI analysis engine |
| **Frontend Dev** | React UI, 3D canvas integration (see `fashion-frontend/`) |
| **AI/ML Lead** | Colab pipeline, `run_pipeline_local.py`, model research |
| **Integration** | `combined_app.html`, API wiring, deployment |

---

## 📋 Phase 1 — Completed

- [x] AI pipeline: body analysis, garment synthesis, SSIM/PSNR metrics
- [x] 3D mannequin with real GLB model and OrbitControls
- [x] Full Flask REST API with SQLite persistence
- [x] Integrated frontend (upload → analyze → try-on → save)
- [x] Zero-dependency fallback server
- [x] Google Colab notebook for AI demo

## 🔮 Phase 2 — Planned

- [ ] Real ML model inference (CatVTON / DCI-VTON)
- [ ] React frontend fully connected to Flask API
- [ ] User authentication (JWT)
- [ ] Cloud deployment (Render / Railway)
- [ ] Texture mapping on 3D avatar from actual garment images

---

## ⚡ Alternative Launch (No Flask required)

```powershell
py partner_backend_5000.py
```
Opens **http://127.0.0.1:5000** automatically — no pip installs needed beyond Python stdlib.

---

*Built with ❤️ for the AI Virtual Fashion project — Phase 1 Complete*
