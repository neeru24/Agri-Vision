import pytest
import io
import os
from PIL import Image
import numpy as np

os.environ.setdefault("SECRET_KEY", "test-secret")
# Avoid reusing a local sqlite file whose schema may lag behind models.py
# (create_all skips existing tables, which breaks inserts after ORM changes).
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

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


@pytest.fixture(autouse=True)
def _stub_magic_mime_when_unavailable(monkeypatch):
    """CI/Linux usually has libmagic; Windows dev boxes often do not. Keep /api/analyze tests runnable."""
    import security_utils as _su

    if _su.magic is not None:
        yield
        return

    def _fallback_mime(sample: bytes) -> str:
        if len(sample) >= 8 and sample.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if len(sample) >= 3 and sample[:3] == b"\xff\xd8\xff":
            return "image/jpeg"
        if len(sample) >= 6 and sample[:6] in (b"GIF87a", b"GIF89a"):
            return "image/gif"
        raise _su.UploadValidationError("Invalid image content.", status_code=400)

    monkeypatch.setattr(_su, "detect_mime_type", _fallback_mime)
    yield


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


@pytest.fixture(autouse=True)
def _disable_rate_limiter_for_most_tests(request):
    """Many tests POST /api/analyze; Flask-Limiter keys by IP → shared 127.0.0.1 hits 429 on CI.

    Keep the limiter enabled only for ``test_api_analyze_rate_limit``, which asserts 429 behavior.
    """
    if request.node.name == "test_api_analyze_rate_limit":
        yield
        return
    limiter = app_module.limiter
    previous = limiter.enabled
    limiter.enabled = False
    try:
        yield
    finally:
        limiter.enabled = previous


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
