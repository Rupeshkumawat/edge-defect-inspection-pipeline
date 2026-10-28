import pytest
import numpy as np
import cv2
from src.pipeline import letterbox, preprocess

def test_letterbox_dimensions():
    # Create dummy 1080x1920 image
    dummy_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    boxed, ratio, (dw, dh) = letterbox(dummy_img, (640, 640))
    
    assert boxed.shape == (640, 640, 3)
    assert ratio == 640 / 1920

def test_preprocess_tensor_shape():
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".jpg", dummy_img)
    tensor, ratio, pad, orig_shape = preprocess(encoded.tobytes())

    assert tensor.shape == (3, 640, 640)
    assert tensor.dtype == np.float32
    assert tensor.max() <= 1.0 and tensor.min() >= 0.0
