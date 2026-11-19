# 👗 Virtual Try-On 3D Studio

> **Virtual fashion try-on & styling prototype** combining an offline-capable Three.js 3D mannequin viewer, Flask + SQLite persistence, and a computer-vision body & style analysis pipeline.



## ✨ Core Features

| Feature | Description |
|---|---|
| 📸 **Photo Upload & CV Analysis** | Upload a photo to estimate body proportions, face shape, skin undertone, and color palette recommendations (`vision_analyzer.py`) |
| 👕 **Clothing Catalog** | Browse garments by category (`tops`, `bottoms`, `dresses`, `outerwear`, `accessories`) |
| 🧍 **Offline 3D Avatar Viewer** | Local Three.js (`r128`) mannequin (`web_app/models/realistic_human.glb`) with `OrbitControls` — works with no internet connection |
| 💾 **Save Outfits** | Persist and reload outfit combinations in SQLite (`instance/stylemate.db`) |
| 🖼️ **2D Try-On Pipeline Demo** | Standalone 5-stage local pipeline (`run_pipeline_local.py`) with cloth-agnostic masking and SSIM/PSNR evaluation |

## 🚀 Quick Start

### 1. Install dependencies
```powershell
py -m pip install -r requirements.txt
```

### 2. Run the main application
```powershell
py mini.py
```
The server starts on **https://www.google.com/search?q=http://127.0.0.1:5000**, initializes `instance/stylemate.db`, and serves `combined_app.html`.

### 3. Run automated smoke tests
```powershell
pytest test_app.py -v
```

> **Offline / Zero-Dependency Fallback:** If you cannot install packages on a demo machine, run `py partner_backend_5000.py` to launch the Python stdlib server on port `5000`.

## 📁 Project Structure

```text
virtual-tryon-3d/
├── mini.py                         # Main Flask + SQLAlchemy backend (port 5000)
├── vision_analyzer.py              # MediaPipe Pose/FaceMesh + Pillow fallback analyzer
├── run_pipeline_local.py           # Local 2D try-on & SSIM/PSNR demo script
├── combined_app.html               # Primary integrated web frontend
├── test_app.py                     # Pytest API & upload smoke tests
├── requirements.txt                # Python dependencies
├── partner_backend_5000.py         # Zero-dependency stdlib fallback server
├── server.py                       # Lightweight static server (port 8000)
├── samples/                        # Sample person, garment, mask, and output images
├── web_app/
│   ├── index.html                  # Standalone 3D Studio viewer
│   ├── js/                         # Local Three.js r128, OrbitControls, GLTFLoader
│   └── models/realistic_human.glb  # 3D human mannequin asset
└── Phase1_AI_Model_Pipeline.ipynb  # Google Colab notebook for GPU VTON experiments
```

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serves `combined_app.html` |
| `GET` | `/api/status` | Health check and database counts |
| `POST` | `/api/users` | Create or fetch a user |
| `GET` | `/api/users` | List registered users |
| `POST` | `/api/upload` | Upload photo (`JPG`/`PNG`) and run analysis |
| `POST` | `/api/analyze` | Run analysis and store an `Analysis` record |
| `GET` | `/api/clothing` | List clothing catalog items |
| `POST` | `/api/outfits/save` | Save an outfit for a user |
| `GET` | `/api/outfits/<user_id>` | Retrieve saved outfits |
| `POST` | `/api/try-on` | Return 3D visualization & try-on payload |

## 🧪 Demo Mode vs. GPU Inference Mode

* **Local CPU Mode (`mini.py` + `vision_analyzer.py` + `run_pipeline_local.py`):** Runs entirely offline on a standard laptop. Body/face landmarks use MediaPipe Pose & FaceMesh (with a Pillow color/aspect fallback), and 2D try-on uses local mask blending for fast live demos.
* **GPU / Colab Mode (`Phase1_AI_Model_Pipeline.ipynb`):** Designed for diffusion-based virtual try-on (e.g., CatVTON / DCI-VTON) when a CUDA GPU is available.

## 🔮 Next Roadmap (Phase 2)

- [ ] Connect `/api/try-on` to a hosted GPU FastAPI/Gradio VTON endpoint
- [ ] Add separate `.glb` garment meshes under `web_app/models/`
- [ ] Add SMPL body-shape scaling and WebXR AR camera try-on
