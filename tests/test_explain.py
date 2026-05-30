import io
import json
import base64
import hashlib
import numpy as np
import pytest
import torch
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
    import numpy as np
    pixels = np.random.randint(50, 200, (300, 300, 3), dtype=np.uint8)
    pixels[50:250, 50:250] = np.random.randint(30, 150, (200, 200, 3), dtype=np.uint8)
    pixels[100:200, 100:200] = np.random.randint(60, 180, (100, 100, 3), dtype=np.uint8)
    pixels[120:180, 120:180] = np.random.randint(10, 100, (60, 60, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)
    return img_byte_arr


class MiniResNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer4 = torch.nn.Sequential(torch.nn.Conv2d(3, 4, kernel_size=3, padding=1))
        self.fc = torch.nn.Linear(4, len(app.disease_classes))

    def forward(self, x):
        x = self.layer4(x)
        x = torch.mean(x, dim=(2, 3))
        return self.fc(x)

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

def test_grad_cam_cache(monkeypatch):
    # Clear cache first
    app.GRAD_CAM_CACHE.clear()
    model = MiniResNet()
    monkeypatch.setattr(app.model_manager, "resnet_model", model)
    monkeypatch.setattr(app.model_manager, "yolo_model", None)
    monkeypatch.setattr(app.model_manager, "loaded", True)
    
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


def test_grad_cam_failure_does_not_break_prediction(monkeypatch):
    app.GRAD_CAM_CACHE.clear()

    def fail_gradcam(*args, **kwargs):
        raise RuntimeError("forced gradcam failure")

    model = MiniResNet()
    monkeypatch.setattr(app.model_manager, "resnet_model", model)
    monkeypatch.setattr(app.model_manager, "yolo_model", None)
    monkeypatch.setattr(app.model_manager, "loaded", True)
    monkeypatch.setattr(app, "generate_gradcam_explanation", fail_gradcam)

    res = app.analyze_image(np.zeros((64, 64, 3), dtype=np.uint8))
    assert "disease" in res
    assert res["disease"]["predicted_class"] in app.disease_classes
    assert res["explainability"]["available"] is False
    assert res["explainability"]["status"] == "failed"

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
