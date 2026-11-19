"""
test_app.py - Comprehensive Unit & Integration Tests for StyleMate v2
Run via: py -m pytest test_app.py -v
"""

import os
import io
import pytest
from app import app, db, User, Clothing, SavedOutfit
from vision_analyzer import analyze_person_image, classify_body_type, get_clothing_size

SAMPLE_IMG_PATH = os.path.join(os.path.dirname(__file__), "samples", "1_target_person.jpg")


@pytest.fixture
def client():
    app.config["TESTING"] = True
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    with app.app_context():
        db.create_all()
        db.session.add(User(name="Test User", email="test@stylemate.ai"))
        db.session.add(Clothing(name="Test Jacket", category="outerwear", color="#111111", model_tag="test"))
        db.session.commit()
        yield app.test_client()
        db.drop_all()


def test_status_endpoint(client):
    res = client.get("/api/status")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "online"
    assert "users_count" in data
    assert "clothing_count" in data


def test_clothing_catalog(client):
    res = client.get("/api/clothing")
    assert res.status_code == 200
    items = res.get_json()
    assert len(items) >= 1
    assert items[0]["name"] == "Test Jacket"


def test_add_clothing(client):
    res = client.post("/api/clothing", json={
        "name": "Summer Linen Shirt",
        "category": "tops",
        "color": "#ffffff"
    })
    assert res.status_code == 201
    data = res.get_json()
    assert "id" in data


def test_try_on_endpoint(client):
    res = client.post("/api/try-on", json={"clothing_id": 1})
    assert res.status_code == 200
    data = res.get_json()
    assert data["clothing_name"] == "Test Jacket"
    assert "texture_url" in data


def test_outfit_save_and_retrieve(client):
    post_res = client.post("/api/outfits", json={
        "user_id": 1,
        "clothing_id": 1,
        "clothing_name": "Test Jacket",
        "clothing_color": "#111111",
        "accessory": "Gold Aviators"
    })
    assert post_res.status_code == 201
    outfit_id = post_res.get_json()["outfit_id"]

    get_res = client.get("/api/outfits/1")
    assert get_res.status_code == 200
    outfits = get_res.get_json()
    assert len(outfits) == 1
    assert outfits[0]["outfit_id"] == outfit_id
    assert outfits[0]["accessory"] == "Gold Aviators"


def test_vision_analyzer_classification():
    body_type = classify_body_type(shoulder_cm=52.0, waist_cm=80.0, hip_cm=95.0)
    assert "Inverted" in body_type or "Athletic" in body_type

    size_m = get_clothing_size(100.0)
    assert size_m == "M"

    size_s = get_clothing_size(92.0)
    assert size_s == "S"


def test_vision_analyzer_on_sample_image():
    if os.path.exists(SAMPLE_IMG_PATH):
        with open(SAMPLE_IMG_PATH, "rb") as f:
            img_bytes = f.read()
        res = analyze_person_image(img_bytes, user_height_cm=175.0)
        assert res["success"] is True
        assert "shoulder_cm" in res["body_measurements"]
        assert "clothing_size" in res
        assert len(res["recommended_clothes"]) > 0


def test_api_upload_endpoint(client):
    if os.path.exists(SAMPLE_IMG_PATH):
        with open(SAMPLE_IMG_PATH, "rb") as f:
            img_bytes = f.read()
        data = {
            "image": (io.BytesIO(img_bytes), "test_person.jpg"),
            "height_cm": "180"
        }
        res = client.post("/api/upload", data=data, content_type="multipart/form-data")
        assert res.status_code == 200
        res_data = res.get_json()
        assert res_data["success"] is True
        assert "clothing_size" in res_data
        assert res_data["body_measurements"]["height_cm"] == 180.0
