# StyleMate — Virtual Try-On 3D

**An AI fashion stylist that turns one full-body photo into a size, a matched catalog and a 3D fitting room.**

Team project · November 2026

| Member | Role |
|---|---|
| Akshit Kandari | Team leader · recommendation engine · AI vision |
| Abhay Singh | Front-end |
| Nikhil Panwar | Back-end |
| Divyansh Kumar | 3D model · integration |

---

## 1. The problem

Online clothing has high return rates, mostly because shoppers cannot tell their size or how a garment will fit.
Size charts are generic, product photos show someone else's body, and recommendations ignore the shopper's build and colouring.

## 2. What StyleMate does

1. **Analyze a photo:**
   - age and gender detection
   - height from a printed marker (or the height you enter)
   - body measurements: chest, waist, hip, shoulder, arm, inseam
   - a size per garment type (e.g. tops M, bottoms S, or kidswear 5-6Y)
   - face shape with frame advice, and skin undertone with a colour palette
   - a confidence score with reasons
2. **Recommend from a catalog:** 312 products ranked by an AI fashion model (FashionCLIP).
   - Matched to audience (men / women / boys / girls), colouring, build and occasion.
   - Filters for budget and "in stock in my size".
   - Every item lists why it was picked and which size to buy.
   - Also: text search, similar items and "complete the look".
3. **Try on in 3D:**
   - an avatar reshaped to your measurements, including child proportions
   - layered outfits: top, jacket, bottom, dress, shoes and five accessory types
   - simulated cloth for skirts, dresses and kurtas that moves with wind
   - a fit heat-map showing where a size is tight, good or loose
4. **Add your own clothes:** upload a garment photo and cloth analysis identifies it.
   - Akshit's VisionFit classifier and FashionCLIP both vote on the type. Agreement is shown, and disagreement is flagged for the user to check.
   - The colour comes from the photo's pixels, and fabric and style come from FashionCLIP. Everything stays editable before saving.
   - The garment joins the catalog (search, recommendations, similar items) and goes onto the 3D avatar with its colour and fabric.
5. **Save outfits** per user in a local database.

Everything runs locally on a laptop; photos never leave the machine.

## 3. Architecture

```text
Browser  (combined_app.html + static/js/avatar_studio.js, three.js)
   │  REST / JSON
   ▼
Flask backend  (mini.py, port 5000)
   ├─ SQLite: users, analyses, 312-product catalog, saved outfits
   ├─ vision_analyzer.py + face_analyzer.py  (MediaPipe, local fallback)
   ├─ fit.py  (size charts, per-product fit, stock-aware size choice)
   └─ ai_services.py ──HTTP──► AI services (services/, Python 3.12, GPU)
                                ├─ 5101  age & gender        MiVOLO v2
                                ├─ 5102  height & measures   MediaPipe pose + ArUco marker
                                ├─ 5103  recommendations     FashionCLIP (tags, search, ranking)
                                └─ 5104  cloth analysis      VisionFit-AI clothing classifier
                                5100  test bench (try each model, start/stop services)
```

If an AI service is off, the app falls back to local analysis and rule-based recommendations, and says so in the UI.

## 4. Tech stack

| Layer | Technology |
|---|---|
| Front-end | HTML/CSS/JavaScript single-page app, three.js r128 (GLTFLoader, OrbitControls) |
| 3D | Ready Player Me avatar (67-bone rig), custom garment shaders, verlet cloth simulation |
| Back-end | Python, Flask, Flask-SQLAlchemy, SQLite |
| Computer vision | MediaPipe Pose, Face Landmarker and segmentation; OpenCV ArUco |
| AI models | MiVOLO v2 (age/gender), FashionCLIP (fashion embeddings), PyTorch, FastAPI services |
| Testing | pytest, headless-browser end-to-end test, automated 3D outfit matrix |

## 5. Roles and work done

### Akshit Kandari — Team leader, recommendation engine, AI vision
- Led the team and the project direction.
- **AI vision:**
  - age and gender detection (MiVOLO v2, face + body)
  - body landmarks and measurements (MediaPipe)
  - height from a printed ArUco marker
  - face-shape analysis with frame advice: the rules come from his VisionFit-AI work, ported to the current MediaPipe and recalibrated on 240 faces
  - skin undertone and colour palette
- **Recommendation engine:**
  - FashionCLIP ranking by occasion and liked items
  - undertone colour palette and body-shape rules
  - budget and size-stock filters
  - per-item reasons and size advice
  - search, similar items, "complete the look"
- **Cloth analysis:** his VisionFit-AI clothing classifier (MobileNetV2, 10 garment types) runs as the 5104 service and reads every uploaded garment. It runs unchanged on Keras 3 with the torch backend.
- **Model evaluation:** benchmarked the candidate models (results in section 6).

### Abhay Singh — Front-end
- **Single-page UI:**
  - photo upload (drag and drop) with height, gender and marker inputs
  - profile card: age, height source, sizes per garment type, measurements, confidence, face shape, palette
  - product cards: price, brand, size advice, reasons
  - occasion chips, search, budget and size filters
  - outfit bar, saved outfits, the "Wearing" panel and fit panel
- Live AI-service status strip.
- Safe rendering of all text, so item names can't inject code into the page.

### Nikhil Panwar — Back-end
- **Flask REST API** (see `README.md`): users, analysis, recommendations, catalog CRUD with validation, search, fit, outfits.
- **SQLite schema:** in-place migration of existing databases.
- **Catalog:** 312 products with brand, price, stock, fit type, pattern and a per-product size chart.
- **Fit engine** (`fit.py`): size charts, cut-aware fit, stock-aware size choice, kidswear by height.
- **Graceful fallback** when AI services are offline; catalog sync with the AI service.
- **Backend test suite:** 32 tests, run against a temporary database with mocked AI services.

### Divyansh Kumar — 3D model, integration
- **3D studio** (`static/js/avatar_studio.js`):
  - avatar reshaped from measurements by bone scaling, with child proportions
  - outfit slots with dressing rules; jackets as an open-front layer
  - belt and tucked shirts, short sleeves and shorts cut to skin
  - procedural accessories (sunglasses, cap, watch, handbag, backpack)
  - verlet cloth for skirts, dresses and kurtas, with leg, hip, arm and hand collisions
  - fit heat-map
  - true-colour lighting
- **Integration:**
  - the AI model services and the client that calls them
  - the test bench
  - the launcher (`START.bat`)
  - removing the photo try-on experiment
- **Testing:**
  - end-to-end browser test (`services/eval/e2e_ui.py`, 21 steps)
  - page smoke test
  - 3D outfit matrix (every catalog item)
  - finding and fixing the issues these turned up

## 6. Results (measured in this project)

| What | Result | How measured |
|---|---|---|
| Garment type recognition | 78.1% FashionCLIP vs 59.9% plain CLIP | 434 labelled product photos |
| FashionCLIP vs a trained classifier | 89.8% vs 62.0% (MobileNetV2) | 400 photos, 10 classes |
| Similar-item search | 73.3% of top-5 match the garment type | 200 catalog items |
| Age estimate | 3.0 years average error; gender 96.3% | 240 faces with known ages |
| Height with marker | +2% (112.2 cm vs 110 cm) | Hand-measured photo, 15 cm marker |
| Height from depth AI | about 2.5× off, so rejected | Depth-Anything-V2 without camera data |
| Face shape | face found in 240 of 240; all six shapes occur | Recalibrated thresholds |
| Analysis time | 1–3 s per photo | RTX 2050 laptop GPU |
| 3D outfits | 319 items + 9 swap sequences, 0 failures, no cloth inside the body | Automated outfit matrix |
| Full user journey | 21 of 21 steps, no JavaScript errors, with or without AI services | Headless-browser test |
| Backend | 32 of 32 tests pass | pytest |

## 7. Work timeline

1. **Phase 1:** Flask + SQLite backend, MediaPipe body analysis, Three.js mannequin viewer, a 2D try-on pipeline demo.
2. **Research:** surveyed open-source models for try-on, 3D bodies, cloth, age, height and recommendations.
3. **Model services:** stood up and benchmarked MiVOLO, height estimation, FashionCLIP and CatVTON as separate services, with a browser test bench.
4. **Integration:**
   - analysis pipeline with fallbacks
   - honest confidence scores, replacing hard-coded values
   - kidswear sizing
   - the 312-product catalog with AI-ranked recommendations
5. **3D studio:**
   - body shaping and cloth simulation
   - outfit slots, layering and accessories
   - clipping fixes
   - garment arrangement: belt, tucking, cap, backpack, colours
6. **Photo try-on (CatVTON):** evaluated, then removed. It was slow and washed out colours, needed the whole GPU and has a non-commercial licence.
7. **VisionFit-AI merge:** face-shape detection and retail catalog fields (price, brand, stock, size chart).
8. **Final test run:** end-to-end, offline-mode and page tests; issues found and fixed.

## 8. How to run

```powershell
py -m pip install -r requirements.txt
START.bat            # starts the AI services (if services\.venv exists), the test bench and the app
# app: http://127.0.0.1:5000    test bench: http://localhost:5100
py -m pytest test_app.py -q
```

Setup of the AI services (Python 3.12, PyTorch) is in `services/README.md`. For accurate height, print `static/marker/aruco_marker_15cm.pdf` at 100% and hold it flat against the chest.

## 9. Limitations and future work

- **Catalog data:** images are 60×80 px thumbnails, and prices and stock are generated demo values. *Next:* real retailer data and HD photos.
- **One base 3D outfit mesh,** and a male base avatar. *Next:* a mesh per garment type, plus female and child base models.
- **Measurements are estimates** from a single photo and are best with the marker. Face-shape thresholds are calibrated without ground-truth labels.
- **Licences:** MiVOLO is non-commercial (CC-BY-NC-4.0). VisionFit-AI has no licence file yet.
- **Next:** mobile camera flow, a personalised 3D body from the photo, real size charts per brand.
