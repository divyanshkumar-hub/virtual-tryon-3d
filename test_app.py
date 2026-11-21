"""Tests for the StyleMate backend (mini.py), the AI-service integration and vision_analyzer.

The suite runs against a throw-away SQLite DB and upload folder, with the AI
services disabled (AI_SERVICES_DISABLED=1). Service behaviour is simulated
with monkeypatch so the integration paths are covered without a GPU.
"""

import io
import os
import sqlite3
import sys
import tempfile

import pytest
from PIL import Image

_TMP = tempfile.mkdtemp(prefix="stylemate_test_")
os.environ["STYLEMATE_DB"] = os.path.join(_TMP, "test.db")
os.environ["STYLEMATE_UPLOADS"] = os.path.join(_TMP, "uploads")
os.environ["AI_SERVICES_DISABLED"] = "1"
os.environ["NO_BROWSER"] = "1"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vision_analyzer import analyze_image_bytes, get_kids_size, measurement_confidence  # noqa: E402


def _png(width: int = 256, height: int = 384) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=(210, 170, 145)).save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------- vision_analyzer
def test_vision_analyzer_returns_valid_structure():
    result = analyze_image_bytes(_png())
    assert result["image_width"] == 256
    assert result["image_height"] == 384
    assert result["body_type"] and result["face_shape"] and result["skin_tone"]
    assert result["undertone"] in {"Warm", "Cool", "Neutral"}
    assert len(result["recommended_colors"]) >= 3
    assert "ssim_score" not in result  # the old hard-coded metric is gone


def test_vision_analyzer_rejects_empty_bytes():
    with pytest.raises(ValueError):
        analyze_image_bytes(b"")


def test_confidence_is_low_without_a_person():
    result = analyze_image_bytes(_png())
    assert result["measurement_confidence"]["level"] == "low"
    assert result["fit_confidence"] <= 30


def test_child_gets_kids_sizing():
    result = analyze_image_bytes(_png(), user_height_cm=112, age=6)
    assert result["size_system"] == "kids"
    assert result["clothing_size"] == "5-6Y"
    assert result["body_type"] == "Child build"


def test_kids_size_table_and_confidence_reasons():
    assert get_kids_size(90) == "18-24M"
    assert get_kids_size(170) == "14Y+"
    c = measurement_confidence("mediapipe_real", {"pose_visibility": 0.9, "full_body_visible": True}, "marker")
    assert c["level"] == "high" and any("marker" in r for r in c["reasons"])


# ---------------------------------------------------------------- Flask app
@pytest.fixture()
def mini_mod():
    import mini

    mini.app.config["TESTING"] = True
    return mini


@pytest.fixture()
def client(mini_mod):
    with mini_mod.app.test_client() as test_client:
        yield test_client


def _analyze(client, endpoint="/api/analyze", **form):
    return client.post(endpoint, data={"image": (io.BytesIO(_png()), "person.png"), **form},
                       content_type="multipart/form-data")


def test_api_status_and_clothing(client):
    res = client.get("/api/status")
    assert res.status_code == 200 and res.get_json()["status"] == "ONLINE"
    items = client.get("/api/clothing").get_json()
    assert isinstance(items, list) and len(items) >= 6
    assert {"audience", "fabric", "material", "image_path"} <= set(items[0])


def test_catalog_seeded_from_json(client):
    kids = client.get("/api/clothing?audience=boys,girls").get_json()
    assert kids, "demo catalog should include kidswear"
    assert all(i["audience"] in ("boys", "girls") for i in kids)


def test_services_endpoint_reports_disabled(client):
    data = client.get("/api/services").get_json()
    assert data["disabled"] is True
    assert set(data["services"]) == {"age", "height", "fashion", "cloth"}


def test_api_upload_and_try_on(client):
    upload_res = _analyze(client, "/api/upload", user_id="1")
    assert upload_res.status_code == 201
    data = upload_res.get_json()
    for key in ("body_measurements", "body_type", "face_shape", "recommended_clothes",
                "recommended_shoes", "recommended_sunglasses", "catalog_recommendations"):
        assert key in data
    clothing = client.get("/api/clothing").get_json()
    params = client.post("/api/try-on", json={"clothing_id": clothing[0]["id"]}).get_json()
    assert "clothing_name" in params and "material" in params and "article_type" in params


def test_api_analyze_success_offline_fallback(client):
    res = _analyze(client)
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    for k in ("shoulder_cm", "chest_cm", "waist_cm", "hip_cm"):
        assert k in data["body_measurements"]
    assert data["recommendation_engine"] == "rules"
    assert data["services_used"] == []
    assert data["height_source"] == "estimated"  # no height entered, no marker, no age service
    assert any("No height given" in n for n in data["notes"])
    assert data["recommended_clothes"] and data["recommended_shoes"] and data["recommended_sunglasses"]
    assert "clothing_size" in data


def test_user_height_and_gender_are_used(client):
    data = _analyze(client, height_cm="182", gender="female").get_json()
    assert data["height_source"] == "user" and data["height_cm"] == 182
    assert data["gender"] == "female" and data["gender_source"] == "user"
    auds = {i["audience"] for g in data["catalog_recommendations"].values() for i in g}
    assert auds <= {"women", "unisex"}


def test_child_detected_by_age_service_gets_kidswear(client, mini_mod, monkeypatch):
    monkeypatch.setattr(mini_mod.ai, "estimate_age", lambda b: {
        "mivolo": {"age": 6.2, "gender": "male", "gender_p": 0.99}, "age_range": [2, 10], "face_box": [10, 10, 60, 70]})
    data = _analyze(client).get_json()
    assert data["is_child"] is True and data["size_system"] == "kids"
    assert data["height_source"] == "estimated" and data["height_cm"] == 116.0  # median for a 6-year-old
    auds = {i["audience"] for g in data["catalog_recommendations"].values() for i in g}
    assert auds == {"boys"}
    assert "dresses" not in data["catalog_recommendations"]


def test_marker_height_overrides_user_height(client, mini_mod, monkeypatch):
    monkeypatch.setattr(mini_mod.ai, "estimate_height", lambda b, h, m: {
        "methods": {"aruco": {"height_cm": 171.5}}, "full_body_visible": True,
        "measurements_cm": {"inseam_approx": 78.0, "arm_length": 59.0}})
    data = _analyze(client, height_cm="160", marker_cm="15").get_json()
    assert data["height_source"] == "marker" and data["height_cm"] == 171.5
    assert data["body_measurements"]["inseam_cm"] == 78.0
    assert any("using the marker" in n for n in data["notes"])


def test_recommendations_use_ai_scores_when_available(client, mini_mod, monkeypatch):
    with mini_mod.app.app_context():
        target = mini_mod.Clothing.query.filter_by(audience="women", category="tops").first()
    monkeypatch.setattr(mini_mod, "ensure_catalog_synced", lambda: True)
    monkeypatch.setattr(mini_mod.ai, "recommend", lambda *a, **k: [{"id": str(target.id), "score": 5.0}])
    res = client.post("/api/recommendations", json={"occasion": "party", "gender": "female"}).get_json()
    assert res["engine"] == "fashionclip"
    assert res["groups"]["tops"][0]["id"] == target.id
    assert "AI style match" in res["groups"]["tops"][0]["reasons"]


def test_search_similar_and_complete_fall_back_to_rules(client):
    s = client.get("/api/search?q=black+shoes").get_json()
    assert s["engine"] == "keyword" and s["results"]
    item_id = s["results"][0]["id"]
    sim = client.get(f"/api/clothing/{item_id}/similar").get_json()
    assert sim["engine"] == "rules"
    comp = client.get(f"/api/clothing/{item_id}/complete").get_json()
    base_cat = client.get(f"/api/clothing/{item_id}").get_json()["category"]
    assert base_cat not in comp["groups"]


def test_photo_tryon_is_gone_and_tagging_reports_offline(client):
    analysis = _analyze(client).get_json()
    top = client.get("/api/clothing?category=tops").get_json()[0]
    r = client.post("/api/try-on/render", json={"analysis_id": analysis["analysis_id"], "clothing_id": top["id"]})
    assert r.status_code in (404, 405)   # photo try-on was removed; 3D try-on runs in the browser
    t = client.post("/api/clothing/tag", data={"image": (io.BytesIO(_png()), "g.png")}, content_type="multipart/form-data")
    assert t.status_code == 503


def test_save_list_delete_outfit(client):
    item = client.get("/api/clothing").get_json()[0]
    saved = client.post("/api/outfits/save", json={"user_id": 1, "clothing_id": item["id"]}).get_json()
    outfits = client.get("/api/outfits/1").get_json()
    assert any(o["outfit_id"] == saved["outfit_id"] and o["image_path"] for o in outfits)
    assert client.delete(f"/api/outfits/item/{saved['outfit_id']}").status_code == 200
    assert not any(o["outfit_id"] == saved["outfit_id"] for o in client.get("/api/outfits/1").get_json())


def test_api_analyze_rejects_empty(client):
    res = client.post("/api/analyze", data={})
    assert res.status_code == 400 and "error" in res.get_json()


def test_api_upload_rejects_empty_and_bad_type(client):
    res = client.post("/api/upload", data={"user_id": "1", "image": (io.BytesIO(b""), "empty.png")},
                      content_type="multipart/form-data")
    assert res.status_code == 400
    res = client.post("/api/upload", data={"user_id": "1", "image": (io.BytesIO(b"GIF89a..."), "x.gif")},
                      content_type="multipart/form-data")
    assert res.status_code == 400


def test_api_try_on_with_clothing_name(client):
    data = client.post("/api/try-on", json={"clothing_name": "Jeans"}).get_json()
    assert "clothing_id" in data and "clothing_name" in data


def test_migration_adds_v2_columns_to_old_database(mini_mod):
    """An old (v1) stylemate.db gains the new columns without losing rows."""
    path = os.path.join(_TMP, "old.db")
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE clothing (id INTEGER PRIMARY KEY, name VARCHAR(100) NOT NULL, category VARCHAR(50) NOT NULL,
                               color VARCHAR(50), image_path VARCHAR(255), model_path VARCHAR(255));
        INSERT INTO clothing (name, category) VALUES ('Old Blazer', 'outerwear');
        CREATE TABLE analysis (id INTEGER PRIMARY KEY, user_id INTEGER, image_path VARCHAR(255),
                               body_measurements TEXT, face_shape VARCHAR(50));
    """)
    con.commit()
    con.close()
    from sqlalchemy import create_engine, inspect

    engine = create_engine(f"sqlite:///{path}")
    mini_mod.migrate_schema(engine)
    cols = {c["name"] for c in inspect(engine).get_columns("clothing")}
    assert {"sku", "audience", "fabric"} <= cols
    with engine.connect() as c:
        assert c.exec_driver_sql("SELECT name FROM clothing").scalar() == "Old Blazer"


# ---------------------------------------------------------------- garment fit
def test_fit_report_sizes_and_verdicts():
    from fit import fit_report

    body = {"chest_cm": 98.7, "waist_cm": 81.3, "hip_cm": 96.7}
    rec = fit_report(body, "tops", "men")
    assert rec["recommended_size"] == "M" and rec["size"] == "M"
    small = fit_report(body, "tops", "men", "XS")
    assert any(r["verdict"] == "tight" for r in small["regions"])
    big = fit_report(body, "tops", "men", "XXL")
    assert any(r["verdict"] == "loose" for r in big["regions"])
    assert fit_report(body, "footwear", "men")["supported"] is False


def test_fit_report_kids_follow_height_and_never_size_down():
    from fit import fit_report

    kid = {"chest_cm": 56.7, "waist_cm": 47.5, "hip_cm": 46.4, "height_cm": 112.2}
    for cat in ("tops", "bottoms", "dresses"):
        r = fit_report(kid, cat, "boys", is_child=True)
        assert r["recommended_size"] in ("5-6Y", "6-7Y")
        assert r["sizes"][0] == "2-3Y"


def test_fit_endpoint(client):
    analysis = _analyze(client, height_cm="178", gender="male").get_json()
    shirt = client.get("/api/clothing?category=tops&audience=men").get_json()[0]
    d = client.post("/api/fit", json={"analysis_id": analysis["analysis_id"], "clothing_id": shirt["id"]}).get_json()
    assert d["supported"] and d["size"] == d["recommended_size"] and d["regions"]
    d2 = client.post("/api/fit", json={"analysis_id": analysis["analysis_id"], "clothing_id": shirt["id"], "size": "XS"}).get_json()
    assert d2["size"] == "XS"
    assert client.post("/api/fit", json={}).status_code == 400


# ---------------------------------------------------------------- face shape (ported from VisionFit-AI)
def test_face_shape_rules_cover_every_shape():
    from face_analyzer import classify_face_shape

    # (width/height, jaw/width, forehead/width)
    assert classify_face_shape(1 / 1.30, 0.82, 0.72) == "Long / Rectangle"
    assert classify_face_shape(1 / 1.18, 0.78, 0.75) == "Heart"
    assert classify_face_shape(1 / 1.18, 0.78, 0.69) == "Diamond"
    assert classify_face_shape(1 / 1.08, 0.86, 0.72) == "Square"
    assert classify_face_shape(1 / 1.08, 0.81, 0.72) == "Round"
    assert classify_face_shape(1 / 1.18, 0.82, 0.72) == "Oval"
    assert classify_face_shape(0, 0.8, 0.7) == "Unclassified"


def test_face_analysis_on_blank_image_reports_not_detected():
    result = analyze_image_bytes(_png())
    assert result["face_analysis"]["face_detected"] is False
    assert result["face_shape"] == "Not detected"


def test_analysis_response_includes_face_fields(client):
    data = _analyze(client).get_json()
    assert "face_analysis" in data and "recommended_frames" in data


# ---------------------------------------------------------------- catalog retail fields
def test_catalog_has_retail_fields_and_filters(client):
    items = client.get("/api/clothing?category=tops&audience=men").get_json()
    it = items[0]
    assert it["brand"] and it["price"] and it["price_is_demo"] is True and it["available_sizes"]
    cheap = client.get("/api/clothing?max_price=500").get_json()
    assert cheap and all(i["price"] <= 500 for i in cheap)
    in_m = client.get("/api/clothing?category=tops&audience=men&size=M").get_json()
    assert all("M" in i["available_sizes"] for i in in_m)
    detail = client.get(f"/api/clothing/{it['id']}").get_json()
    assert detail["size_chart"] and set(next(iter(detail["size_chart"].values()))) <= {"chest", "waist", "hip"}


def test_catalog_patch_validates_and_updates(client):
    it = client.get("/api/clothing?category=tops&audience=men").get_json()[0]
    assert client.patch(f"/api/clothing/{it['id']}", json={"price": -5}).status_code == 400
    assert client.patch(f"/api/clothing/{it['id']}", json={"available_sizes": "M"}).status_code == 400
    assert client.patch(f"/api/clothing/{it['id']}", json={"fit_type": "baggy-ish"}).status_code == 400
    assert client.patch(f"/api/clothing/{it['id']}", json={"size_chart": {"M": {"chest": "big"}}}).status_code == 400
    r = client.patch(f"/api/clothing/{it['id']}", json={"price": 1499, "fit_type": "slim", "available_sizes": ["S", "M"],
                                                       "size_chart": {"S": {"chest": 96, "waist": 90}, "M": {"chest": 102, "waist": 96}}})
    assert r.status_code == 200
    body = r.get_json()["item"]
    assert body["price"] == 1499 and body["fit_type"] == "slim" and body["available_sizes"] == ["S", "M"]
    assert client.patch("/api/clothing/999999", json={"price": 1}).status_code == 404


def test_recommendations_budget_and_size_filters(client):
    analysis = _analyze(client, height_cm="178", gender="male").get_json()
    r = client.post("/api/recommendations", json={"analysis_id": analysis["analysis_id"], "max_price": 800}).get_json()
    prices = [i["price"] for g in r["groups"].values() for i in g if i["price"] is not None]
    assert prices and max(prices) <= 800 and r["filtered_out"]["budget"] > 0
    r2 = client.post("/api/recommendations", json={"analysis_id": analysis["analysis_id"], "only_my_size": True}).get_json()
    advised = [i["size_advice"] for g in r2["groups"].values() for i in g if i.get("size_advice")]
    assert advised and all(a["in_stock"] for a in advised)
    assert client.post("/api/recommendations", json={"max_price": "cheap"}).status_code == 400


def test_fit_uses_product_chart_cut_and_stock():
    from fit import fit_report, product_size_chart

    body = {"chest_cm": 98.7, "waist_cm": 81.3, "hip_cm": 96.7}
    for cut in ("slim", "regular", "relaxed", "oversized"):
        chart = product_size_chart("tops", "men", ["S", "M", "L"], cut)
        rep = fit_report(body, "tops", "men", chart=chart, available=["S", "M", "L"], fit_type=cut)
        assert rep["recommended_size"] == "M" and rep["uses_product_chart"]   # buy your size; the cut is the style
    rep = fit_report(body, "tops", "men", available=["XL", "XXL"])
    assert rep["recommended_available"] is False and rep["closest_available"] == "XL" and "not in stock" in rep["summary"]


def test_sizes_by_category_and_gender_note(client):
    data = _analyze(client, height_cm="175").get_json()
    assert set(data["sizes_by_category"]) >= {"tops", "bottoms"}
    assert data["clothing_size"] == data["sizes_by_category"]["tops"]
    assert any("Gender could not be detected" in n for n in data["notes"])
    female = _analyze(client, height_cm="165", gender="female").get_json()
    assert "dresses" in female["sizes_by_category"]
    assert not any("Gender could not be detected" in n for n in female["notes"])


# ---------------------------------------------------------------- clothing upload (cloth analysis)
def _garment_png(colour=(180, 20, 30)) -> bytes:
    """A coloured garment-sized block on a white background."""
    img = Image.new("RGB", (200, 240), color=(250, 250, 250))
    img.paste(Image.new("RGB", (120, 170), color=colour), (40, 35))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _post_garment(client, endpoint, **form):
    return client.post(endpoint, data={"image": (io.BytesIO(_garment_png()), "g.png"), **form},
                       content_type="multipart/form-data")


def test_garment_colour_ignores_the_background(mini_mod):
    c = mini_mod.garment_colour(_garment_png((30, 45, 120)))
    rgb = tuple(int(c["hex"][i:i + 2], 16) for i in (1, 3, 5))
    assert max(abs(a - b) for a, b in zip(rgb, (30, 45, 120))) < 20
    assert c["name"] == "Navy Blue"
    assert mini_mod.garment_colour(b"not an image") is None


def test_clothing_analyze_offline_still_suggests_fields(client):
    d = _post_garment(client, "/api/clothing/analyze").get_json()
    assert d["visionfit"] is None and d["fashionclip"] is None and d["engines"] == [] and d["agree"] is None
    assert d["suggested"]["color"].startswith("#") and d["suggested"]["colour_name"] == "Red"
    assert {"key": "shoes", "label": "Shoes / sneakers", "category": "footwear"} in d["types"]
    assert "denim" in d["fabrics"]
    bad = client.post("/api/clothing/analyze", data={"image": (io.BytesIO(b"xx"), "g.png")}, content_type="multipart/form-data")
    assert bad.status_code == 400


def _fc_tags(type_label):
    return {"type": [{"label": type_label, "p": 0.8}], "colour": [{"label": "red", "p": 0.9}],   # FashionCLIP /tag shape
            "fabric": [{"label": "denim", "p": 0.7}], "style": [{"label": "casual", "p": 0.6}]}


def test_clothing_analyze_combines_visionfit_and_fashionclip(client, mini_mod, monkeypatch):
    monkeypatch.setattr(mini_mod.ai, "classify_cloth", lambda b: {"category": "pants", "score": 0.7,
                                                                  "predictions": [{"label": "pants", "score": 0.7}]})
    monkeypatch.setattr(mini_mod.ai, "tag_garment", lambda b: _fc_tags("jeans"))
    d = _post_garment(client, "/api/clothing/analyze").get_json()
    assert d["agree"] is True and d["suggested"]["type"] == "jeans"   # agreement keeps FashionCLIP's finer type
    assert d["suggested"]["fabric"] == "denim" and d["suggested"]["name"] == "My red jeans"

    monkeypatch.setattr(mini_mod.ai, "tag_garment", lambda b: _fc_tags("sneakers"))
    d = _post_garment(client, "/api/clothing/analyze").get_json()
    assert d["agree"] is False and d["suggested"]["type"] == "shoes"

    monkeypatch.setattr(mini_mod.ai, "tag_garment", lambda b: None)
    d = _post_garment(client, "/api/clothing/analyze").get_json()
    assert d["agree"] is None and d["suggested"]["type"] == "trousers" and d["engines"] == ["visionfit"]


def test_clothing_upload_adds_wearable_item_then_delete(client, mini_mod):
    r = _post_garment(client, "/api/clothing/upload", type="sandals", name="My red sandals", color="#B41E1E",
                      colour_name="Red", fabric="leather", usage="Casual", audience="women", price="999")
    assert r.status_code == 201
    item = r.get_json()["item"]
    assert item["category"] == "footwear" and item["article_type"] == "Sandals"   # wearable in the shoes slot
    assert item["color"] == "#b41e1e" and item["fabric"] == "leather" and item["price"] == 999
    path = os.path.join(mini_mod.USER_GARMENTS_DIR, os.path.basename(item["image_path"]))
    assert os.path.isfile(path)
    mine = client.get("/api/clothing?source=user").get_json()
    assert mine[0]["id"] == item["id"] and all(i["sku"].startswith("user-") for i in mine)

    client.post("/api/outfits/save", json={"user_id": 1, "clothing_id": item["id"]})
    assert client.delete(f"/api/clothing/{item['id']}").status_code == 200
    assert not os.path.exists(path)
    assert client.get(f"/api/clothing/{item['id']}").status_code == 404
    assert all(o["clothing_id"] != item["id"] for o in client.get("/api/outfits/1").get_json())

    catalog_item = client.get("/api/clothing?category=tops").get_json()[0]
    assert client.delete(f"/api/clothing/{catalog_item['id']}").status_code == 403


def test_clothing_upload_validates_fields(client):
    ok = dict(type="tshirt", name="Tee", color="#112233", fabric="cotton")
    assert _post_garment(client, "/api/clothing/upload", **{**ok, "type": "spacesuit"}).status_code == 400
    assert _post_garment(client, "/api/clothing/upload", **{**ok, "name": " "}).status_code == 400
    assert _post_garment(client, "/api/clothing/upload", **{**ok, "color": "red"}).status_code == 400
    assert _post_garment(client, "/api/clothing/upload", **{**ok, "fabric": "kevlar"}).status_code == 400
    assert _post_garment(client, "/api/clothing/upload", **{**ok, "price": "-5"}).status_code == 400
    assert _post_garment(client, "/api/clothing/upload", **{**ok, "audience": "cats"}).status_code == 400


def test_tag_add_puts_shoes_in_footwear_with_a_colour(client, mini_mod, monkeypatch):
    monkeypatch.setattr(mini_mod.ai, "tag_garment", lambda b: _fc_tags("sneakers"))
    item = _post_garment(client, "/api/clothing/tag", add="1").get_json()["item"]
    try:
        assert item["category"] == "footwear" and item["color"] and item["color"].startswith("#")
    finally:
        client.delete(f"/api/clothing/{item['id']}")
