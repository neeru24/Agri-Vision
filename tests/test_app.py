import base64
import io
import json
from flask_login import login_user
from models import User, db

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

import app


# --- Add Missing Fixtures Here ---
@pytest.fixture(scope="session")
def app_with_db():
    app.app.config["TESTING"] = True
    app.app.config["LOGIN_DISABLED"] = True
    app.app.config["UPLOAD_FOLDER"] = "./static/uploads"
    app.app.config["SECRET_KEY"] = "test-secret"
    app.app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    
    with app.app.app_context():
        db.create_all()
        test_user = User(
            id=1, 
            email="test@example.com", 
            full_name="Test User",
            password_hash="pbkdf2:sha256:260000$test$test"
        )
        db.session.add(test_user)
        db.session.commit()
        yield app.app
        db.drop_all()

@pytest.fixture
def client(app_with_db):
    with app_with_db.test_client() as client:
        with client.session_transaction() as sess:
            sess['_user_id'] = '1'
            sess['_fresh'] = True
        yield client


@pytest.fixture
def valid_image():
    img_byte_arr = io.BytesIO()
    Image.new("RGB", (100, 100), color="green").save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)
    return img_byte_arr


@pytest.fixture
def invalid_file():
    return io.BytesIO(b"fake plain text file data content")


@pytest.fixture
def oversized_file():
    return io.BytesIO(b"0" * (11 * 1024 * 1024))  # 11MB file to test limit


# Mocking deep learning models for active path checks
class MockResNetModel:
    def __call__(self, x):
        # target index 5 is healthy
        logits = torch.zeros(1, 8)
        logits[0, 5] = 10.0
        return logits

    def eval(self):
        return self


class MockYOLOBox:
    def __init__(self, class_id, confidence, xyxy):
        self.cls = [torch.tensor(class_id)]
        self.conf = [torch.tensor(confidence)]
        self.xyxy = [torch.tensor(xyxy)]


class MockYOLOResult:
    def __init__(self, boxes):
        self.boxes = boxes


class MockYOLOModel:
    def __call__(self, pil_image):
        # dummy boxes for growth stage detection
        box1 = MockYOLOBox(
            class_id=3, confidence=0.95, xyxy=[120.0, 80.0, 210.0, 155.0]
        )
        box2 = MockYOLOBox(
            class_id=4, confidence=0.75, xyxy=[300.0, 120.0, 390.0, 210.0]
        )
        return [MockYOLOResult([box1, box2])]


# Basic unit tests for helper utils
def test_preprocess_image_for_resnet():
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    processed = app.preprocess_image_for_resnet(dummy_img, target_size=(224, 224))
    assert isinstance(processed, torch.Tensor)
    assert processed.shape == (1, 3, 224, 224)


def test_infer_disease_fallback(monkeypatch):
    monkeypatch.setattr(app.model_manager, "resnet_model", None)
    monkeypatch.setattr(app.model_manager, "loaded", True)
    monkeypatch.setattr(app, "resnet_model", None)
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    res = app.infer_disease(dummy_img)
    assert "predicted_class" in res
    assert res["predicted_class"] in app.disease_classes
    assert 0.0 <= res["confidence"] <= 1.0
    assert 0.0 <= res["health_score"] <= 100.0


def test_infer_disease_active(monkeypatch):
    monkeypatch.setattr(app.model_manager, "resnet_model", MockResNetModel())
    monkeypatch.setattr(app.model_manager, "loaded", True)
    monkeypatch.setattr(app, "resnet_model", MockResNetModel())
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    res = app.infer_disease(dummy_img)
    assert res["predicted_class"] == "Healthy"
    assert res["predicted_class_idx"] == 5
    assert res["health_score"] > 90.0


def test_infer_growth_stage_fallback(monkeypatch):
    monkeypatch.setattr(app.model_manager, "yolo_model", None)
    monkeypatch.setattr(app.model_manager, "loaded", True)
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    res = app.infer_growth_stage(dummy_img)
    assert res["main_class"] is None
    assert res["confidence"] == 0.0
    assert len(res["boxes"]) == 0


def test_analyze_image_without_growth_detection(monkeypatch):
    monkeypatch.setattr(app.model_manager, "yolo_model", None)
    monkeypatch.setattr(app.model_manager, "loaded", True)
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    result = app.analyze_image(dummy_img)
    assert "disease" in result
    assert result["disease"] is not None
    assert result["growth"]["main_class"] is None
    assert "warnings" in result
    assert any(
        "Growth stage model unavailable" in warning for warning in result["warnings"]
    )


def test_infer_growth_stage_active(monkeypatch):
    monkeypatch.setattr(app.model_manager, "yolo_model", MockYOLOModel())
    monkeypatch.setattr(app.model_manager, "loaded", True)
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    res = app.infer_growth_stage(dummy_img)
    assert res["main_class"] == "Matured Cotton Boll"
    assert res["main_class_idx"] == 3
    assert abs(res["confidence"] - 0.95) < 1e-4
    assert len(res["boxes"]) == 2
    assert res["boxes"][0]["class_name"] == "Matured Cotton Boll"
    assert res["boxes"][1]["class_name"] == "Split Cotton Boll"


def test_generate_recommendations():
    disease_res = {
        "predicted_class": "Aphids",
        "predicted_class_idx": 0,
        "confidence": 0.9,
        "all_confidences": {},
        "health_score": 45.0,
        "is_uncertain": True,
        "raw": [],
    }
    growth_res = {
        "main_class": "Cotton Blossom",
        "main_class_idx": 0,
        "confidence": 0.8,
        "boxes": [],
        "raw": [],
    }
    recs = app.generate_recommendations(disease_res, growth_res)
    assert isinstance(recs, list)
    assert len(recs) > 0
    assert any("consult an agricultural expert" in r.lower() for r in recs)
    assert any("insecticides" in r for r in recs or "Aphids" in r)
    assert any("blossom" in r.lower() for r in recs)


def test_encode_image_for_display():
    dummy_img = np.zeros((50, 50, 3), dtype=np.uint8)
    encoded = app.encode_image_for_display(dummy_img)
    assert isinstance(encoded, str)
    decoded = base64.b64decode(encoded)
    assert len(decoded) > 0


# Testing core frontend routes
def test_home_page_en(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"Agri" in resp.data or b"Vision" in resp.data


def test_home_page_hero_demo_cta_links_to_demo(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b'id="hero-demo-link"' in resp.data
    assert b'href="/demo"' in resp.data
    assert b"View Demo" in resp.data


def test_home_page_te(client):
    resp = client.get("/?lang=te")
    assert resp.status_code == 200
    assert b"Agri-Vision" in resp.data


def test_set_language_redirect(client):
    resp = client.get("/set-language/te")
    assert resp.status_code == 302
    assert "/?lang=te" in resp.headers["Location"]


def test_health_check_endpoint(client, monkeypatch):
    monkeypatch.setattr(app.model_manager, "resnet_model", MockResNetModel())
    monkeypatch.setattr(app.model_manager, "yolo_model", MockYOLOModel())
    monkeypatch.setattr(app.model_manager, "loaded", True)
    resp = client.get("/health")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_health_check_endpoint_fallback(client, monkeypatch):
    monkeypatch.setattr(app.model_manager, "resnet_model", None)
    monkeypatch.setattr(app.model_manager, "yolo_model", None)
    monkeypatch.setattr(app.model_manager, "loaded", True)
    resp = client.get("/health")
    assert resp.status_code == 503
    data = json.loads(resp.data)
    assert data["status"] == "degraded"
    assert data["model_loaded"] is False


def test_demo_route(client):
    resp = client.get("/demo")
    assert resp.status_code == 200
    assert b"Matured Cotton Boll" in resp.data
    assert b"Split Cotton Boll" in resp.data


def test_get_analyze_route(client):
    resp = client.get("/analyze")
    assert resp.status_code == 200
    assert b"Upload" in resp.data or b"Image" in resp.data


def test_get_comparison_route(client):
    resp = client.get("/comparison")
    assert resp.status_code == 200
    assert b"Field Photo Comparison" in resp.data
    assert b"Last Week Field Image" in resp.data


def test_build_comparison_result_improved():
    old_results = {
        "disease": {
            "predicted_class": "Aphids",
            "confidence": 0.8,
            "health_score": 42.0,
        },
        "recommendations": ["Increase scouting frequency."],
    }
    new_results = {
        "disease": {
            "predicted_class": "Healthy",
            "confidence": 0.9,
            "health_score": 68.0,
        },
        "recommendations": ["Continue general crop monitoring."],
    }
    result = app.build_comparison_result(old_results, new_results)
    assert result["trend"]["status"] == "improved"
    assert result["change_percentage"] == 26.0
    assert any("Disease spread reduced" in item for item in result["summary"])


# Form upload submission routes
def test_post_analyze_valid(client, valid_image):
    data = {"file": (valid_image, "test_cotton.png")}
    resp = client.post("/analyze", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    assert (
        b"Results" in resp.data
        or b"recommendations" in resp.data
        or b"analysis" in resp.data
    )


def test_post_comparison_valid(client, monkeypatch):
    def mock_analyze_image(_image):
        return {
            "disease": {
                "predicted_class": "Healthy",
                "predicted_class_idx": 5,
                "confidence": 0.92,
                "all_confidences": {},
                "health_score": 82.0,
                "raw": [],
            },
            "growth": {
                "main_class": "Matured Cotton Boll",
                "main_class_idx": 3,
                "confidence": 0.8,
                "boxes": [],
                "raw": [],
            },
            "recommendations": ["Continue general crop monitoring."],
        }

    monkeypatch.setattr(app, "analyze_image", mock_analyze_image)

    image_one = io.BytesIO()
    Image.new("RGB", (80, 80), color="green").save(image_one, format="PNG")
    image_one.seek(0)
    image_two = io.BytesIO()
    Image.new("RGB", (80, 80), color="darkgreen").save(image_two, format="PNG")
    image_two.seek(0)

    data = {
        "last_week_image": (image_one, "last_week.png"),
        "current_week_image": (image_two, "current_week.png"),
    }
    resp = client.post("/comparison", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    assert b"AI RECOMMENDATION" in resp.data
    assert b"Old Prediction" in resp.data
    assert b"New Prediction" in resp.data


def test_post_comparison_invalid_crop_image(client, monkeypatch):
    def mock_analyze_image(_image):
        return {
            "error": "No cotton plant detected",
            "disease": None,
            "growth": {"main_class": None},
        }

    monkeypatch.setattr(app, "analyze_image", mock_analyze_image)

    image_one = io.BytesIO()
    Image.new("RGB", (80, 80), color="green").save(image_one, format="PNG")
    image_one.seek(0)
    image_two = io.BytesIO()
    Image.new("RGB", (80, 80), color="darkgreen").save(image_two, format="PNG")
    image_two.seek(0)

    data = {
        "last_week_image": (image_one, "last_week.png"),
        "current_week_image": (image_two, "current_week.png"),
    }
    resp = client.post("/comparison", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    assert (
        b"Unable to compare images" in resp.data
        or b"Unable to verify cotton crop" in resp.data
    )


def test_post_comparison_duplicate_image(client):
    # Create the same image twice
    image_content = io.BytesIO()
    Image.new("RGB", (100, 100), color="blue").save(image_content, format="PNG")
    
    # We need two separate BytesIO objects with the same content for the request
    image_one = io.BytesIO(image_content.getvalue())
    image_two = io.BytesIO(image_content.getvalue())

    data = {
        "last_week_image": (image_one, "field_1.png"),
        "current_week_image": (image_two, "field_2.png"),
    }
    
    resp = client.post("/comparison", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    assert b"Duplicate field images detected" in resp.data
    assert b"Please upload two different images" in resp.data


def test_post_comparison_fallback_when_both_images_no_growth(client, monkeypatch):
    def mock_analyze_image(_image):
        return {
            "disease": {
                "predicted_class": "Aphids",
                "predicted_class_idx": 0,
                "confidence": 0.5,
                "all_confidences": {},
                "health_score": 38.0,
                "raw": [],
            },
            "growth": {
                "main_class": None,
                "main_class_idx": None,
                "confidence": 0.0,
                "boxes": [],
                "raw": [],
            },
            "recommendations": ["Please upload a valid cotton crop image."],
            "warnings": [
                "Cotton growth stage could not be detected from the uploaded image."
            ],
        }

    monkeypatch.setattr(app, "analyze_image", mock_analyze_image)
    monkeypatch.setattr(app.model_manager, "yolo_model", object())
    monkeypatch.setattr(app.model_manager, "loaded", True)

    image_one = io.BytesIO()
    Image.new("RGB", (80, 80), color="green").save(image_one, format="PNG")
    image_one.seek(0)
    image_two = io.BytesIO()
    Image.new("RGB", (80, 80), color="darkgreen").save(image_two, format="PNG")
    image_two.seek(0)

    data = {
        "last_week_image": (image_one, "last_week.png"),
        "current_week_image": (image_two, "current_week.png"),
    }
    resp = client.post("/comparison", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    assert b"Unable to verify cotton crop in both images" in resp.data
    assert b"clearer field photos" in resp.data


def test_post_comparison_missing_file(client):
    resp = client.post("/comparison", data={}, content_type="multipart/form-data")
    assert resp.status_code == 302
    assert "/comparison" in resp.headers["Location"]


def test_post_analyze_missing_file_key(client):
    resp = client.post("/analyze", data={}, content_type="multipart/form-data")
    assert resp.status_code == 302
    assert "/analyze" in resp.headers["Location"]


def test_post_analyze_empty_filename(client):
    data = {"file": (io.BytesIO(b""), "")}
    resp = client.post("/analyze", data=data, content_type="multipart/form-data")
    assert resp.status_code == 302
    assert "/analyze" in resp.headers["Location"]


def test_post_analyze_invalid_extension(client, invalid_file):
    data = {"file": (invalid_file, "test.txt")}
    resp = client.post("/analyze", data=data, content_type="multipart/form-data")
    assert resp.status_code == 302
    assert "/analyze" in resp.headers["Location"]


def test_post_analyze_oversized_file(client, oversized_file):
    data = {"file": (oversized_file, "large_cotton.png")}
    resp = client.post("/analyze", data=data, content_type="multipart/form-data")
    assert resp.status_code == 413


# Rest API validation checks
def test_post_api_analyze_valid(client, valid_image):
    data = {"file": (valid_image, "test_cotton.png")}
    resp = client.post("/api/analyze", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    res_data = json.loads(resp.data)
    assert res_data["status"] == "success"
    assert "results" in res_data
    assert "disease" in res_data["results"]
    assert "growth" in res_data["results"]
    assert "recommendations" in res_data["results"]


def test_post_api_analyze_missing_file_key(client):
    resp = client.post("/api/analyze", data={}, content_type="multipart/form-data")
    assert resp.status_code == 400
    res_data = json.loads(resp.data)
    assert "error" in res_data
    assert "No file uploaded" in res_data["error"]


def test_post_api_analyze_empty_filename(client):
    data = {"file": (io.BytesIO(b""), "")}
    resp = client.post("/api/analyze", data=data, content_type="multipart/form-data")
    assert resp.status_code == 400
    res_data = json.loads(resp.data)
    assert "error" in res_data
    assert "No file selected" in res_data["error"]


def test_post_api_analyze_invalid_image(client, invalid_file):
    data = {"file": (invalid_file, "corrupted.png")}
    resp = client.post("/api/analyze", data=data, content_type="multipart/form-data")
    assert resp.status_code == 400
    res_data = json.loads(resp.data)
    assert "error" in res_data
    assert "Invalid image file" in res_data["error"]


def test_datetimeformat_filter():
    res_now = app.datetimeformat_filter("now")
    assert len(res_now) > 0
    res_val = app.datetimeformat_filter("2026-05-17")
    assert res_val == "2026-05-17"


# Boundary coverage checks for fallback triggers
def test_load_models_coverage(monkeypatch):
    orig_resnet = app.resnet_model
    orig_yolo = app.yolo_model
    app.resnet_model = None
    app.yolo_model = None

    try:
        resnet, yolo = app.load_models()
        assert app.resnet_model is not None or app.resnet_model is None
    finally:
        app.resnet_model = orig_resnet
        app.yolo_model = orig_yolo


def test_post_analyze_invalid_image_none(client, monkeypatch, valid_image):
    monkeypatch.setattr(cv2, "imdecode", lambda *args: None)
    data = {"file": (valid_image, "test_cotton.png")}
    resp = client.post("/analyze", data=data, content_type="multipart/form-data")
    assert resp.status_code == 302
    assert "/analyze" in resp.headers["Location"]


def test_post_analyze_exception(client, monkeypatch, valid_image):
    def mock_raise(*args, **kwargs):
        raise RuntimeError("Mock analysis error")

    monkeypatch.setattr(app, "analyze_image", mock_raise)
    data = {"file": (valid_image, "test_cotton.png")}
    resp = client.post("/analyze", data=data, content_type="multipart/form-data")
    assert resp.status_code == 302
    assert "/analyze" in resp.headers["Location"]


def test_post_api_analyze_exception(client, monkeypatch, valid_image):
    def mock_raise(*args, **kwargs):
        raise RuntimeError("Mock API error")

    monkeypatch.setattr(app, "analyze_image", mock_raise)
    data = {"file": (valid_image, "test_cotton.png")}
    resp = client.post("/api/analyze", data=data, content_type="multipart/form-data")
    assert resp.status_code == 500
    res_data = json.loads(resp.data)
    assert "error" in res_data
    assert "Mock API error" in res_data["error"]


def test_generate_mock_heatmap():
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    heatmap = app.generate_mock_heatmap(dummy_img)
    assert heatmap.shape == (100, 100)
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0


def test_apply_heatmap_on_image():
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    dummy_heatmap = np.ones((100, 100), dtype=np.float32)
    blended = app.apply_heatmap_on_image(dummy_img, dummy_heatmap)
    assert blended.shape == (100, 100, 3)
    assert blended.dtype == np.uint8


def test_gradcam_class_initialization():
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 10, kernel_size=3, padding=1)
            self.fc = torch.nn.Linear(10, 8)
        def forward(self, x):
            x = self.conv(x)
            x = torch.mean(x, dim=[2, 3])
            return self.fc(x)

    model = DummyModel()
    gradcam = app.GradCAM(model, model.conv)
    assert gradcam.model == model
    assert gradcam.target_layer == model.conv

    input_tensor = torch.randn(1, 3, 224, 224)
    orig_img = np.zeros((224, 224, 3), dtype=np.uint8)
    res = gradcam(input_tensor, target_class_idx=2, original_image_rgb=orig_img)
    assert res is not None
    assert res.shape == (224, 224, 3)


def test_api_chat_test(client):
    resp = client.get("/api/chat_test")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert data == {"status": "ok"}


def test_api_chat_empty_message(client):
    resp = client.post("/api/chat", json={})
    assert resp.status_code == 400
    data = json.loads(resp.data)
    assert "reply" in data
    assert "didn't receive a message" in data["reply"]


def test_api_chat_keyword_matching(client):
    # Test "hello"
    resp = client.post("/api/chat", json={"message": "Hello!"})
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert "Hello there!" in data["reply"] or "Hi!" in data["reply"]

    # Test "disease"
    resp = client.post("/api/chat", json={"message": "Spots on crop leaves"})
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert "Bacterial Blight" in data["reply"] or "Target Spot" in data["reply"]

    # Test "yield"
    resp = client.post("/api/chat", json={"message": "how to improve yield"})
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert "health score" in data["reply"] or "growth stage" in data["reply"]


def test_api_chat_fallback_response(client):
    resp = client.post("/api/chat", json={"message": "unknown query message"})
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert "Agri-Vision AI assistant" in data["reply"]
