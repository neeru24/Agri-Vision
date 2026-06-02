import pytest
import io
import os
from PIL import Image
import numpy as np

os.environ.setdefault("SECRET_KEY", "test-secret")

import app as app_module

app_module.resnet_model = app_module.model_manager.resnet_model
app_module.yolo_model = app_module.model_manager.yolo_model


def _load_models_for_legacy_tests():
    resnet_model, yolo_model = app_module.model_manager.load_models()
    app_module.resnet_model = resnet_model
    app_module.yolo_model = yolo_model
    return resnet_model, yolo_model


app_module.load_models = _load_models_for_legacy_tests

flask_app = app_module.app

@pytest.fixture
def app():
    """Configures the Flask app for testing."""
    flask_app.config.update({
        "TESTING": True,
        "LOGIN_DISABLED": False,
        "MAX_CONTENT_LENGTH": 10 * 1024 * 1024,
        # Max content length is kept at 10MB to test oversized file uploads
    })
    return flask_app


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
def client(app):
    """Provides a Flask test client."""
    return app.test_client()

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
    """Generates a dummy file larger than 10MB to trigger MaxContentLength (MAX_CONTENT_LENGTH = 10 * 1024 * 1024)."""
    # 11MB of dummy data
    file_bytes = io.BytesIO(b"0" * (11 * 1024 * 1024))
    file_bytes.seek(0)
    return file_bytes
