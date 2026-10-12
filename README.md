# V-STYLE 3D: AI Virtual Try-On Studio (Phase 1 Package)

Welcome to the **V-STYLE 3D** mini-project repository! This project has been organized and built specifically for your **Phase 1 Evaluation**.

---

## 🚀 Quick Start (Run in 10 Seconds)

### Option 1: Double-Click (Easiest)
Simply double-click the file:
```
run_project.bat
```
This will start the server and automatically launch the 3D Studio in your default browser at `http://localhost:8000`.

### Option 2: Terminal Command
Open PowerShell or Command Prompt in this folder and run:
```powershell
py server.py
```
*(Zero pip install dependencies required! Uses Python standard library).*

---

## 📁 Project Contents & Structure

| File / Folder | Purpose |
| :--- | :--- |
| **`run_project.bat`** | One-click launcher to run the project on Windows. |
| **`server.py`** | Python backend server handling API requests (`/api/status`, `/api/tryon`, `/api/catalog`). |
| **`run_pipeline_local.py`** | Standalone Python script that executes the complete AI/CV pipeline locally! |
| **`web_app/index.html`** | The complete 3D Fashion Studio web app built with Three.js (Mannequin, Clothes, Accessories, AI panel). |
| **`Phase1_AI_Model_Pipeline.ipynb`** | Google Colab Jupyter Notebook with the end-to-end AI pipeline + Gradio public demo. |
| **`PROJECT_REPORT_PHASE1.md`** | Complete academic report ready for submission/printing for your professors. |
| **`PRESENTATION_SCRIPT.md`** | Step-by-step viva presentation script divided among all 4 team members. |

---

## 🧠 Running the AI Model on Google Colab

1. Open your browser and navigate to: [https://colab.research.google.com/](https://colab.research.google.com/)
2. Click **Upload** -> select `Phase1_AI_Model_Pipeline.ipynb` from this folder (`c:\code\virtual_tryon_3d\`).
3. Click **Runtime** -> **Change runtime type** -> choose **T4 GPU**.
4. Click **Runtime** -> **Run all**.
5. It will run without dependency conflicts, display the 4-panel comparison figure, and give you a public **Gradio Live URL** (e.g. `https://xxxx.gradio.live`) to share with your evaluator!

---

## 👥 Team Work Allocation (For Evaluator Review)

* **Student 1:** AI Model Pipeline & Latent Diffusion Synthesis (`Phase1_AI_Model_Pipeline.ipynb`).
* **Student 2:** Dataset Ingestion & Pose-Agnostic Masking Preprocessing.
* **Student 3:** 3D WebGL Mannequin, Camera Orbit & Accessory Rigging (`web_app/index.html`).
* **Student 4:** Backend API Server, System Integration & Academic Report (`server.py`, `PROJECT_REPORT_PHASE1.md`).

