import pytest
import io
import os
from PIL import Image
import numpy as np

os.environ.setdefault("SECRET_KEY", "test-secret")

import app as app_module
import models
from models import db, User

# Standard mocks for ML model behavior (without shadowing packages)
class MockResNetModel:
    def __call__(self, x):
        # target index 5 is healthy
        import torch
        logits = torch.zeros(1, 8)
        logits[0, 5] = 10.0
        return logits
    def eval(self):
        return self

class MockYOLOBox:
    def __init__(self, class_id, confidence, xyxy):
        import torch
        self.cls = [torch.tensor(class_id)]
        self.conf = [torch.tensor(confidence)]
        self.xyxy = [torch.tensor(xyxy)]

class MockYOLOResult:
    def __init__(self, boxes):
        self.boxes = boxes

class MockYOLOModel:
    def __call__(self, pil_image):
        import torch
        # dummy boxes for growth stage detection
        box1 = MockYOLOBox(class_id=3, confidence=0.95, xyxy=[120.0, 80.0, 210.0, 155.0])
        box2 = MockYOLOBox(class_id=4, confidence=0.75, xyxy=[300.0, 120.0, 390.0, 210.0])
        return [MockYOLOResult([box1, box2])]

@pytest.fixture(scope="session", autouse=True)
def mock_models():
    """Session-wide fixture to mock the heavy ML models."""
    app_module.resnet_model = MockResNetModel()
    app_module.yolo_model = MockYOLOModel()
    yield
    # Restore original attributes if needed (though session is ending)
    if hasattr(app_module.model_manager, 'resnet_model'):
        app_module.resnet_model = app_module.model_manager.resnet_model
    if hasattr(app_module.model_manager, 'yolo_model'):
        app_module.yolo_model = app_module.model_manager.yolo_model

@pytest.fixture(scope="session")
def app():
    """Configures the Flask app for testing."""
    flask_app = app_module.app
    flask_app.config.update({
        "TESTING": True,
        "LOGIN_DISABLED": True,
        "MAX_CONTENT_LENGTH": 10 * 1024 * 1024,
        "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
        "WTF_CSRF_ENABLED": False
    })
    
    with flask_app.app_context():
        db.create_all()
        # Create a default test user with ID "1"
        if not User.query.get("1"):
            user = User(
                id="1", 
                email="test@example.com", 
                full_name="Test User",
                password_hash="pbkdf2:sha256:260000$test$test"
            )
            db.session.add(user)
            db.session.commit()
    
    return flask_app

@pytest.fixture
def client(app):
    """Provides a Flask test client with an active session for user '1'."""
    with app.test_client() as client:
        with client.session_transaction() as sess:
            sess['_user_id'] = "1"
            sess['_fresh'] = True
        yield client

@pytest.fixture(autouse=True)
def allow_synthetic_test_images(monkeypatch):
    """Keep image-quality heuristics from rejecting generated unit-test PNGs."""
    monkeypatch.setattr(
        app_module,
        "safe_validate_image_quality",
        lambda _image: ({"is_blocking": False, "warnings": []}, False),
        raising=False,
    )

@pytest.fixture
def valid_image():
    """Generates a valid green 100x100 PNG image in-memory."""
    img = Image.new('RGB', (100, 100), color='green')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr

@pytest.fixture
def invalid_file():
    """Generates a dummy text file."""
    file_bytes = io.BytesIO(b"This is just some plain text, not an image file.")
    file_bytes.seek(0)
    return file_bytes

@pytest.fixture
def oversized_file():
    """Generates a dummy file larger than 10MB."""
    file_bytes = io.BytesIO(b"0" * (11 * 1024 * 1024))
    file_bytes.seek(0)
    return file_bytes
