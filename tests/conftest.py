import pytest
import io
import os
import shutil
import tempfile
from pathlib import Path

from PIL import Image
import numpy as np

# Flask-SQLAlchemy 3 binds the engine URI when ``db.init_app`` runs. Updating
# ``SQLALCHEMY_DATABASE_URI`` later does not rebuild the engine, so tests kept
# writing to ``agri_vision.db``. Point at a dedicated SQLite file before import.
_TEST_DB_DIR = tempfile.mkdtemp(prefix="agri_pytest_")
_TEST_DB_PATH = Path(_TEST_DB_DIR) / "session.sqlite3"
os.environ["DATABASE_URL"] = "sqlite:///" + str(_TEST_DB_PATH).replace("\\", "/")

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


@pytest.fixture(scope="session")
def app_with_db():
    """Prepare shared DB tables and seed login user for tests that use ``client``."""
    from models import User, db

    flask_app.config.update(
        {
            "TESTING": True,
            "LOGIN_DISABLED": True,
            "UPLOAD_FOLDER": "./static/uploads",
            "SECRET_KEY": "test-secret",
        }
    )

    with flask_app.app_context():
        db.session.remove()
        db.engine.dispose()
        db.drop_all()
        db.create_all()
        test_user = User(
            id="1",
            email="test@example.com",
            full_name="Test User",
            password_hash="pbkdf2:sha256:260000$test$test",
        )
        db.session.add(test_user)
        db.session.commit()

    yield flask_app

    with flask_app.app_context():
        db.session.remove()
        db.drop_all()
        db.engine.dispose()
    shutil.rmtree(_TEST_DB_DIR, ignore_errors=True)


@pytest.fixture
def app():
    """Configures the Flask app for testing."""
    flask_app.config.update(
        {
            "TESTING": True,
            "LOGIN_DISABLED": False,
            "MAX_CONTENT_LENGTH": 10 * 1024 * 1024,
            # Max content length is kept at 10MB to test oversized file uploads
        }
    )
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
    img = Image.new("RGB", (100, 100), color="green")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
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
