import pytest
import io
from PIL import Image
import numpy as np
from app import app as flask_app

@pytest.fixture
def app():
    """Configures the Flask app for testing."""
    flask_app.config.update({
        "TESTING": True,
        # Max content length is kept at 10MB to test oversized file uploads
    })
    return flask_app

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
