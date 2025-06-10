# tests/test_utils.py (or tests/test_inference_preprocess.py)

import numpy as np
import os
import tempfile
import cv2
import pytest

from machine_learning.src.inference import preprocess_image

def test_preprocess_image_scales_and_shapes(tmp_path):
    # Create a dummy image file on disk to mimic reading from a path
    # e.g., a small 10x10 BGR image
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    # draw something so resize is exercised
    img[:] = [50, 100, 150]
    # write to temp file
    file_path = tmp_path / "test.jpg"
    cv2.imwrite(str(file_path), img)

    # Call preprocess_image
    processed = preprocess_image(str(file_path))
    # preprocess_image resizes to (224,224,3) and scales to [0,1]
    assert processed.shape == (224, 224, 3)
    # Values should be float32 in [0,1]
    assert processed.dtype == np.float32
    assert processed.min() >= 0.0 and processed.max() <= 1.0

def test_preprocess_image_invalid_path():
    with pytest.raises(Exception):
        preprocess_image("non_existent_file.jpg")
