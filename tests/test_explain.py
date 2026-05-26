import io
import json
import base64
import hashlib
import numpy as np
import pytest
from PIL import Image
import cv2

import app

@pytest.fixture
def client():
    app.app.config["TESTING"] = True
    with app.app.test_client() as client:
        yield client

@pytest.fixture
def valid_image():
    img_byte_arr = io.BytesIO()
    Image.new("RGB", (100, 100), color="green").save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)
    return img_byte_arr

def test_generate_pure_heatmap():
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    dummy_heatmap = np.ones((100, 100), dtype=np.float32)
    pure_heatmap = app.generate_pure_heatmap(dummy_img, dummy_heatmap)
    
    assert pure_heatmap.shape == (100, 100, 3)
    assert pure_heatmap.dtype == np.uint8

def test_api_explain_endpoint_valid(client, valid_image):
    data = {"file": (valid_image, "test_cotton.png")}
    resp = client.post("/api/explain", data=data, content_type="multipart/form-data")
    
    assert resp.status_code == 200
    res_data = json.loads(resp.data)
    
    assert res_data["status"] == "success"
    assert "heatmap_b64" in res_data
    assert "heatmap_only_b64" in res_data
    assert res_data["target_layer"] == "ResNet50 layer4[-1]"
    assert "image_b64" in res_data
    assert "predicted_class" in res_data
    assert "confidence" in res_data

def test_api_explain_missing_file(client):
    resp = client.post("/api/explain", data={}, content_type="multipart/form-data")
    assert resp.status_code == 400
    res_data = json.loads(resp.data)
    assert "No file uploaded" in res_data["error"]

def test_api_explain_invalid_extension(client):
    data = {"file": (io.BytesIO(b"dummy text"), "test.txt")}
    resp = client.post("/api/explain", data=data, content_type="multipart/form-data")
    assert resp.status_code == 400
    res_data = json.loads(resp.data)
    assert "Invalid file type" in res_data["error"]

def test_grad_cam_cache():
    # Clear cache first
    app.GRAD_CAM_CACHE.clear()
    
    dummy_img = np.zeros((64, 64, 3), dtype=np.uint8)
    image_hash = hashlib.sha256(dummy_img.tobytes()).hexdigest()
    
    # Run first time - should not be in cache
    assert app.get_cached_grad_cam(image_hash) is None
    
    res1 = app.analyze_image(dummy_img)
    assert image_hash in app.GRAD_CAM_CACHE
    
    # Check that second run returns cached values
    cached_cam = app.get_cached_grad_cam(image_hash)
    assert cached_cam is not None
    
    res2 = app.analyze_image(dummy_img)
    assert res1["grad_cam_image_b64"] == res2["grad_cam_image_b64"]
    assert res1["heatmap_only_b64"] == res2["heatmap_only_b64"]

def test_grad_cam_cache_eviction():
    app.GRAD_CAM_CACHE.clear()
    
    # Fill cache up to MAX_CACHE_SIZE
    max_size = app.MAX_CACHE_SIZE
    for i in range(max_size):
        dummy_img = np.zeros((16, 16, 3), dtype=np.uint8)
        dummy_img[0, 0, 0] = i # unique content
        app.set_cached_grad_cam(str(i), f"overlay_{i}", f"heatmap_{i}")
        
    assert len(app.GRAD_CAM_CACHE) == max_size
    
    # Set one more - should trigger FIFO eviction of the first key "0"
    app.set_cached_grad_cam(str(max_size), "new_overlay", "new_heatmap")
    assert len(app.GRAD_CAM_CACHE) == max_size
    assert "0" not in app.GRAD_CAM_CACHE
    assert str(max_size) in app.GRAD_CAM_CACHE
