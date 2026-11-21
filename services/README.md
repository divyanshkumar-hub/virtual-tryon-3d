# Experimental model services

Standalone FastAPI services for evaluating candidate models before wiring them into `mini.py`.
Each runs on its own port; all accept `multipart/form-data` image uploads. Tested on an RTX 2050 (4 GB).

| Port | Service | Model(s) | License | Verdict |
|---|---|---|---|---|
| 5101 | `age_service` `/estimate` | MiVOLO v2 (face+body), ViT age baseline | **CC-BY-NC-4.0** | ✅ adopt (non-commercial) |
| 5102 | `height_service` `/estimate` | MediaPipe pose-heavy + segmentation, ArUco, Depth-Anything-V2 metric | Apache-2.0 | ✅ marker method; ❌ depth-only |
| 5103 | `fashion_service` `/tag` `/search` `/similar/{id}` `/recommend` `/catalog/add` | FashionCLIP | MIT | ✅ adopt |
| 5104 | `cloth_service` `/classify` | VisionFit-AI clothing classifier by Akshit Kandari (MobileNetV2, 10 classes), Keras 3 on the torch backend, CPU | no licence file yet | ✅ cloth analysis for uploads, alongside FashionCLIP |
| 5113 | same service, `FASHION_CLIP_MODEL=openai/clip-vit-base-patch32` | baseline only | MIT | — |

## Setup (Python 3.12 — torch has no wheels for the system 3.15)

```bash
python -m pip install --user uv
python -m uv venv services/.venv --python 3.12
python -m uv pip install --python services/.venv/Scripts/python.exe torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m uv pip install --python services/.venv/Scripts/python.exe --prerelease=allow -r services/requirements.txt
python -m uv pip install --python services/.venv/Scripts/python.exe --no-deps --no-build-isolation git+https://github.com/WildChlamydia/MiVOLO.git
# MiVOLO needs two import fallbacks in site-packages/mivolo/model/create_timm_model.py
# (remap_checkpoint -> remap_state_dict, split_model_name_tag moved to timm.models._registry)
curl -LO https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task   # -> services/models/
curl -LO https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite  # -> services/models/
services\start_services.bat bench    # 5101-5104 + test bench (http://localhost:5100); add "eval" for the 5113 CLIP baseline
```

## Browser test bench

`http://localhost:5100` (`bench_service.py` + `bench.html`): start/stop services, try each model with samples or your own uploads, catalog search/recommend, try-on, and a one-click **autonomous test** that scores random labelled samples live.

## Evaluation

```bash
services/.venv/Scripts/python services/eval/prepare_data.py     # needs parquet shards in services/eval_data/raw
services/.venv/Scripts/python services/eval/run_evals.py fashion age height
```
Results land in `services/eval_results/*.json`.

### Fashion tagging / retrieval — 600 products from `ashraq/fashion-product-images-small` (60×80 px thumbnails)

| metric | FashionCLIP | OpenAI CLIP | n |
|---|---|---|---|
| garment type top-1 / top-3 | **78.1%** / 96.1% | 59.9% / 95.2% | 434 |
| colour top-1 / top-3 | **61.6%** / 86.8% | 57.0% / 81.2% | 560 |
| style (usage) top-1 / top-3 | **24.7%** / 54.8% | 12.7% / 20.2% | 598 |
| fabric top-1 / top-3 (weak labels from title) | 38.1% / 57.1% | 33.3% / 42.9% | 21 |
| similar item same type, P@5 | **73.3%** | 65.3% | 200 |
| text search "colour + type", P@5 | **54.3%** | 40.4% | 46 |

Main confusions: t-shirt→polo (29), sneakers→formal shoes (15), heels→sandals (14), kurta→dress (8).
Style and fabric are weak. Use them as soft re-ranking signals, not as labels shown to the user.

### Age — 240 faces from UTKFace (stratified, ages 0–9 and 20–49 only in this shard)

| | MiVOLO v2 | ViT baseline |
|---|---|---|
| MAE | **3.0 y** | 3.79 y |
| within ±5 y | **79.6%** | 73.8% |
| MAE 0–12 / 20–39 / 40–59 | 0.74 / 3.17 / 4.93 | 1.07 / 4.01 / 6.07 |
| gender acc. | 96.3% | — |
| latency (GPU) | ~130 ms | |

UTKFace may overlap MiVOLO's training data, so treat these numbers as optimistic. On a real full-body photo of a child it gave 6.2 y from face+body and 8.8 y from body only (plausible). It errs older for 40-somethings (40→56 worst case).

### Height — limited ground truth (one real full-body photo, hand-annotated)

| method | result | truth |
|---|---|---|
| ArUco 15 cm marker, full res | **112.2 cm** (+2.0%) | 110 cm |
| ArUco, 360×480 JPEG q50 | **112.1 cm** | 110 cm |
| person pixel extent from segmentation | 950 px | 939 px annotated (+1.2%) |
| Depth-Anything-V2 metric + 26 mm focal prior | 271–303 cm ❌ | 110 cm |

Monocular metric depth is unusable for absolute height without real camera intrinsics. The image had no EXIF and was probably cropped.
Body-plane measurements at the marker scale look plausible (shoulder 28.6 cm, hip width 13.7 cm for a ~5-year-old).
Validating properly needs 10+ adult full-body photos with tape-measured heights.

### Try-on — CatVTON (evaluated, then removed)

CatVTON ran on the 4 GB GPU at 384×512 with its vitonhd weights in about 19 s per image, or about 140 s at 768×1024.
Garment shape and print transferred, but colours came out washed out in every setting tried (fp16/bf16, both weight sets, larger masks).
It needed the whole GPU and is licensed CC-BY-NC-SA, so it was removed from the project. The 3D avatar is the try-on.

## Known issues found in the main app (`vision_analyzer.py`)
- `fit_confidence` (97.4) and `ssim_score` (0.925) are hard-coded constants, not measurements.
- Shoulder width is clamped to 34–56 cm and hip width to 24–48 cm. A child's 28.6 / 13.7 cm becomes 34 / 24 cm, which gives the wrong body type ("Inverted Triangle") and adult blazer recommendations.
- Height defaults to 175 cm when the user gives none, and every measurement scales from it.
