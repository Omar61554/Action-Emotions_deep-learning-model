"""Small utilities for the project."""

import numpy as np


def normalize_image(img):
    """Normalize image to range 0..1"""
    img = img.astype(np.float32)
    mi = img.min()
    ma = img.max()
    if ma - mi == 0:
        return np.zeros_like(img)
    return (img - mi) / (ma - mi)
