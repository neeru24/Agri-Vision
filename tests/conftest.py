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
        "MAX_CONTENT_LENGTH": 10 * 1024 * 1024,  # 10MB to test oversized file uploads
    })
    return flask_app

@pytest.fixture
def client(app):
    """Provides a Flask test client."""
    return app.test_client()

@pytest.fixture
def valid_image():
    """Generates a valid 300x300 PNG image with sufficient detail to pass quality checks."""
    import numpy as np
    pixels = np.random.randint(50, 200, (300, 300, 3), dtype=np.uint8)
    pixels[50:250, 50:250] = np.random.randint(30, 150, (200, 200, 3), dtype=np.uint8)
    pixels[100:200, 100:200] = np.random.randint(60, 180, (100, 100, 3), dtype=np.uint8)
    pixels[120:180, 120:180] = np.random.randint(10, 100, (60, 60, 3), dtype=np.uint8)
    img = Image.fromarray(pixels, 'RGB')
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
