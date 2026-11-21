# 👗 StyleMate — AI Stylist & 3D Try-On

Upload a full-body photo and StyleMate works out your **age, gender, height, body measurements and clothing size**.
It then recommends **items from a 312-product catalog** matched to your audience (men / women / boys / girls), colouring, build and occasion.
You can try them on a **3D avatar** shaped to your proportions, with layered outfits, simulated cloth and a per-size fit map.

## Architecture

```text
combined_app.html  (UI, Three.js avatar)
        │  REST
        ▼
mini.py  (Flask + SQLite, port 5000) ── ai_services.py ──► optional model services (services/, FastAPI)
  │ vision_analyzer.py (MediaPipe, local)                     5101 age      MiVOLO v2 (face + body)
  │ static/catalog/   (312 products)                          5102 height   MediaPipe pose-heavy + ArUco marker
  └ instance/stylemate.db                                     5103 fashion  FashionCLIP (tags, search, recs)
                                                              5104 cloth    VisionFit clothing classifier (uploads)
                                                              5100 bench    test bench / service control
```

Every AI feature degrades gracefully. If a service is off, the app uses local MediaPipe analysis and rule-based recommendations, and the UI tells you which engine produced each result.

## Run it

```powershell
py -m pip install -r requirements.txt
START.bat                      # starts the AI services (if services\.venv exists) and the app
# or just the app:
py mini.py                     # http://127.0.0.1:5000
```

The AI services need their own Python 3.12 environment with PyTorch. Setup, benchmark results and licences are in [services/README.md](services/README.md).

## What happens on upload (`POST /api/upload`)

1. **Age & gender:** the MiVOLO age service. Gender you enter always wins over the detected one.
2. **Height scale**, best source first:
   - **printed ArUco marker** (4×4 dictionary, side length entered in the UI, ±2% in tests)
   - **height you enter**
   - **estimate** from the detected age and gender, flagged as low confidence
3. **Measurements:** MediaPipe landmarks with limits proportional to stature, so children are not forced into adult ranges. Inseam and arm length come from the heavier pose model.
4. **Size:** adult XS–XXL by chest, or kidswear (e.g. `5-6Y`) by height for under-14s.
5. **Face shape:** `face_analyzer.py` crops the face, using the age service's face box or the pose landmarks, then runs MediaPipe Face Landmarker (478 points). It measures cheek width, face height, jaw width and forehead width and classifies the face as Oval, Round, Square, Heart, Diamond or Long. It then suggests frame styles for that shape. The face crop is also used to sample cheek skin colour for the skin-tone palette.

   The rule structure and landmark ratios are ported from the VisionFit-AI project. Its original thresholds put every face into Diamond, Round or Oval (measured on 240 UTKFace faces), so the cut-offs were recalibrated to population percentiles of those ratios. There is no face-shape ground truth here, so treat the shape as an estimate.
6. **Confidence:** a 0–100 score with reasons, such as landmark visibility, whole body in frame, and where the scale came from. This replaces the old hard-coded "97.4% / SSIM 0.925".
7. **Recommendations:** the catalog is filtered by audience, an optional budget, and optionally "in stock in my size". It is then ranked by:
   - FashionCLIP similarity to the occasion and the items you ♥
   - an undertone colour palette
   - body-shape preferences
   - a penalty for infant sizes shown to older kids

   Each item carries the reasons it was picked, plus `size_advice`: which size of that product fits you, whether it is in stock, and the nearest stocked size.

## Catalog

Each product has a brand, price, available sizes, fit type (slim / regular / relaxed / …), pattern and its **own size chart** (finished chest / waist / hip per size, adjusted for the cut).

- The fit view and size advice use the product's chart and stock. A relaxed cut in your size counts as "good", because the extra room is the style.
- The demo catalog's prices and stock levels are generated (`price_is_demo: true`, INR). The source dataset has none. Real values can be set with `PATCH /api/clothing/<id>`.

## 3D studio (`static/js/avatar_studio.js`)

- **Stance:** one natural standing stance. The rig's A-pose is relaxed to arms-down, palms in. There are no pose animations, only breathing and blinking. The arms open slightly when a skirt, dress, jacket or bag needs room.
- **Outfit slots:** top, jacket, bottom, dress, shoes, sunglasses, watch, cap, belt, bag/backpack. Dressing rules keep every combination consistent:
  - a dress replaces the top and bottom
  - a top or bottom replaces a dress
  - one bag at a time
  - a belt only shows on trousers

  Every change rebuilds the whole look from the slots. The page lists what is worn, and each item can be taken off.
- **Layering:** jackets and blazers are a second skinned layer over the shirt, open at the front with lapels. T-shirts show bare forearms, shorts and capris show bare lower legs, and dresses and skirts show bare legs.
- **Accessories:** sunglasses, cap, watch, handbag and backpack are 3D pieces attached to the head, wrist, hand or back, in the item's colour. The backpack straps run over the shoulders.
- **Belt and tucking:** the belt recolours the belt that is modelled on the trousers, instead of floating a separate ring over the shirt. Shirts are tucked in when a belt is worn, or when a formal shirt goes with formal trousers, so the waistband shows. T-shirts, sweaters and hoodies stay untucked.
- **True colours:** linear tone mapping and moderate lights keep rendered garment colours within about 10% of the catalog colour. ACES tone mapping had over-saturated dark fabrics.
- **Body shaping:** individual bones are scaled from your measurements (shoulders, chest, waist, hips, leg and arm length) instead of stretching the whole model. Children get child proportions: a larger head and smaller hands.
- **Cloth simulation:** skirts, dresses and kurta hems are a verlet cloth pinned at the waist and stepped at a fixed 60 Hz. It collides with the pelvis (an ellipsoid), thighs, shins, the gap between the legs, the forearms and the hands, with a 1.2 cm thickness margin. The pelvis is resolved first, then two limb passes, so no point ends inside the body. Fabric drives stiffness, flare, weight and damping: silk and chiffon fall close, denim and leather stand out. The wind slider moves the cloth.
- **No clipping on fitted garments:** the shirt below the waistband is hidden under skirts, dresses and kurtas, and the trouser waistband is hidden under them too. Hem flutter only moves fabric outward. The fit ease only resizes the torso, never the sleeves, and shrinking is capped at 5 mm.
- **Fit view:** pick a size and the garment loosens or tightens on the body. A heat-map shows tight (red) → good (green) → loose (blue) at chest, waist and hips, next to the recommended size.
- Developer check page: `/static/dev/studio_check.html?items=[...]&wind=0.8` renders any outfit and prints the worn slots, plus the number of cloth points inside the body (expected 0).

## API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/status` | Health and DB counts |
| GET | `/api/services` | Online state of each AI service and catalog sync progress |
| POST | `/api/users` · GET `/api/users` | Create/switch user · list |
| POST | `/api/upload` (strict user) · `/api/analyze` | Photo analysis. Form fields: `image`, `user_id`, optional `height_cm`, `gender`, `marker_cm` |
| GET | `/api/analysis/<id>` | Stored analysis |
| POST | `/api/recommendations` | `{analysis_id, occasion, liked_ids, max_price?, only_my_size?}` → items grouped by category, with reasons and size advice |
| GET | `/api/clothing` (`?category=&audience=&brand=&max_price=&size=&source=user`) · `/api/clothing/<id>` (includes size chart) | Catalog |
| POST | `/api/clothing` · PATCH `/api/clothing/<id>` | Add or edit a product. Price, sizes, size chart and fit type are validated |
| GET | `/api/search?q=&audience=` | Text search (FashionCLIP, keyword fallback) |
| GET | `/api/clothing/<id>/similar` · `/complete` | Similar items · items that go with it |
| POST | `/api/clothing/analyze` | Garment photo → cloth analysis: VisionFit classifier + FashionCLIP votes, colour read from the pixels, and suggested catalog fields |
| POST | `/api/clothing/upload` | Garment photo + confirmed fields (`type`, `name`, `color`, `fabric`, `usage`, `audience`, `price?`) → new catalog item, wearable in 3D |
| DELETE | `/api/clothing/<id>` | Remove a garment you uploaded (catalog products can't be deleted) |
| POST | `/api/clothing/tag` | Garment photo → type / colour / fabric / style (`add=1` adds it to the catalog) |
| POST | `/api/outfits/save` · GET `/api/outfits/<user>` · DELETE `/api/outfits/item/<id>` | Saved outfits |
| POST | `/api/try-on` | 3D avatar parameters for an item: colour, fabric material (roughness / metalness), article type, image |
| POST | `/api/fit` | `{analysis_id, clothing_id, size?}` → your size for that product (from its size chart and cut), stock, and per-region ease with a tight / snug / good / relaxed / loose verdict |

## Tests

```powershell
py -m pytest test_app.py -q                                   # 32 backend tests (temp DB, AI services mocked/disabled)
services\.venv\Scripts\python services\eval\e2e_ui.py           # drives the real page in headless Edge: 21 steps
services\.venv\Scripts\python services\eval\smoke_pages.py URL  # loads pages, reports JS errors / failed requests
# 3D outfit matrix (every catalog item + swap sequences): open /static/dev/studio_check.html?matrix=1
```

`e2e_ui.py` uploads a photo, analyzes it, uses the budget filter, search, similar items and "complete the look". It then wears one item per category in 3D, checks for cloth penetration and the fit panel, saves and deletes an outfit, exercises every control, and fails on any JavaScript error or failed request. Use `--base http://127.0.0.1:5050` to test a second instance, for example with `AI_SERVICES_DISABLED=1`.

## Height marker

Print [`static/marker/aruco_marker_15cm.pdf`](static/marker/aruco_marker_15cm.pdf) at 100% / actual size (the sheet has a 10 cm check ruler). Hold it flat against the chest in the photo. If the printed side isn't exactly 15 cm, enter the measured size in the app.

## Project layout

```text
mini.py                 Flask app, analysis pipeline, recommendation engine, DB migration
ai_services.py          HTTP client for the model services (timeouts, graceful fallback)
vision_analyzer.py      MediaPipe body analysis, sizing, confidence, skin tone
face_analyzer.py        Face Landmarker face shape + frame advice (ported from VisionFit-AI)
face_landmarker.task    MediaPipe face model (3.7 MB)
fit.py                  size charts, product size charts by cut, per-region fit and stock-aware size choice
static/js/avatar_studio.js  3D engine: outfit slots, body shaping, garment shaders, accessories, cloth simulation
combined_app.html       Front end (analysis, catalog, outfit builder, 3D try-on with fit map)
static/catalog/         312 demo products (from the public "Fashion Product Images (Small)" dataset)
scripts/build_catalog.py  rebuilds static/catalog from the dataset parquet
services/               model services, test bench, evaluation scripts and results
web_app/                Three.js r128 + realistic_human.glb avatar
```

## Known limitations

- **Catalog images are 60×80 thumbnails.** That limits FashionCLIP tagging accuracy (type 78%, colour 62%). Higher-resolution product photos are the main upgrade.
- **Height** is accurate with a marker. Monocular depth estimation was tested and rejected (~2.5× off).
- **MiVOLO has a non-commercial licence** (CC-BY-NC-4.0). Photo try-on (CatVTON) was evaluated and removed: blurry, muted colours, it needed the whole 4 GB GPU, and its licence is non-commercial.
- **The 3D avatar** reuses one base outfit mesh: shirts, t-shirts, sweaters and hoodies share a silhouette, recoloured with fabric-specific finish. Jackets are an inflated copy of it. Only skirts, dresses and kurta hems are simulated cloth. Shoes share one mesh, so heels, sandals and sneakers differ only in colour. Per-garment meshes (e.g. from CLO3D, Marvelous Designer or GarmentCode) would be the next step.
- **The avatar's base body is male.** Women and children get proportional reshaping, not a different base model.
