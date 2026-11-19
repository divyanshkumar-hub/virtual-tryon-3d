"""Smoke tests for the Virtual Try-On 3D Flask backend (mini.py) and vision_analyzer."""

import io
import os
import sys

import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vision_analyzer import analyze_image_bytes


def _make_sample_png_bytes(width: int = 256, height: int = 384) -> bytes:
    img = Image.new("RGB", (width, height), color=(210, 170, 145))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_vision_analyzer_returns_valid_structure():
    png_bytes = _make_sample_png_bytes()
    result = analyze_image_bytes(png_bytes)

    assert result["image_width"] == 256
    assert result["image_height"] == 384
    assert result["body_type"]
    assert result["face_shape"]
    assert result["skin_tone"]
    assert result["undertone"] in {"Warm", "Cool", "Neutral"}
    assert isinstance(result["recommended_colors"], list)
    assert len(result["recommended_colors"]) >= 3


def test_vision_analyzer_rejects_empty_bytes():
    with pytest.raises(ValueError):
        analyze_image_bytes(b"")


@pytest.fixture()
def client():
    import mini  # Imported lazily so vision_analyzer tests run even without mini.py in isolation

    mini.app.config["TESTING"] = True
    with mini.app.test_client() as test_client:
        yield test_client


def test_api_status_and_clothing(client):
    res = client.get("/api/status")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "ONLINE"

    clothing_res = client.get("/api/clothing")
    assert clothing_res.status_code == 200
    assert isinstance(clothing_res.get_json(), list)


def test_api_upload_and_try_on(client):
    png_bytes = _make_sample_png_bytes()
    upload_res = client.post(
        "/api/upload",
        data={"user_id": "1", "image": (io.BytesIO(png_bytes), "demo.png")},
        content_type="multipart/form-data",
    )
    assert upload_res.status_code in (200, 201)

    clothing = client.get("/api/clothing").get_json()
    if clothing:
        tryon_res = client.post("/api/try-on", json={"clothing_id": clothing[0]["id"]})
        assert tryon_res.status_code == 200
        assert "clothing_name" in tryon_res.get_json()