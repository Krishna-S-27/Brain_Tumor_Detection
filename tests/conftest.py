import pytest
import numpy as np
from PIL import Image
from io import BytesIO
import os

import sys
# Insert the path to machine_learning/ so that `src` is visible
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'machine_learning'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

@pytest.fixture
def dummy_image():
    """Returns a dummy PIL image for testing"""
    img = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))
    return img

@pytest.fixture
def dummy_image_bytes(dummy_image):
    """Returns dummy image in bytes format (for API testing)"""
    buf = BytesIO()
    dummy_image.save(buf, format='JPEG')
    buf.seek(0)
    return buf

@pytest.fixture
def api_url():
    """Base URL for the FastAPI backend"""
    return "http://localhost:8000"

@pytest.fixture(scope="session", autouse=True)
def prepare_environment():
    """Any pre-test environment setup like directory creation"""
    os.makedirs("tests/tmp", exist_ok=True)
    yield
    # Teardown if needed
    if os.path.exists("tests/tmp"):
        import shutil
        shutil.rmtree("tests/tmp")

@pytest.fixture
def sample_image_path():
    return os.path.join(os.path.dirname(__file__), "sample_image.jpg")
