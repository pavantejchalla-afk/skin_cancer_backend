from __future__ import annotations

from io import BytesIO
import numpy as np
from PIL import Image

ERROR_MESSAGE = "Invalid Image: Strictly only skin lesion images are allowed. Please upload a clear photo of a skin lesion."
IMAGE_SIZE = (224, 224)

# Pre-computed HAM10000 dataset reference statistics
REF_MEAN = np.array([0.027710566, 0.02103821, 0.11911831, 0.6291344], dtype=np.float32)
REF_STD = np.array([0.012806608, 0.007206459, 0.22829008, 0.07139179], dtype=np.float32)


def compute_feature_vector(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    gray = rgb.mean(axis=2)
    hsv = np.asarray(image.convert("HSV"), dtype=np.float32) / 255.0

    skin_ratio = float(
        ((hsv[:, :, 0] >= 0.02) & (hsv[:, :, 0] <= 0.18) & (hsv[:, :, 1] >= 0.12) & (hsv[:, :, 2] >= 0.15)).mean()
    )
    edge_density = float(np.mean(np.abs(np.diff(gray, axis=0))) + np.mean(np.abs(np.diff(gray, axis=1))))
    variance = float(rgb.var())
    luma = float(gray.mean())

    return np.array([variance, edge_density, skin_ratio, luma], dtype=np.float32)


def validate_image_bytes(image_bytes: bytes) -> None:
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image = image.convert("RGB")
            if image.size[0] < 100 or image.size[1] < 100:
                raise ValueError(ERROR_MESSAGE)

            resized = image.resize(IMAGE_SIZE)
            features = compute_feature_vector(resized)
    except Exception as exc:
        raise ValueError(ERROR_MESSAGE) from exc

    variance, edge_density, skin_ratio, luma = features[0], features[1], features[2], features[3]

    # 1. Reject solid colors / dark / overexposed images
    if variance < 0.002 or luma < 0.10 or luma > 0.95:
        raise ValueError(ERROR_MESSAGE)

    # 2. Reject text documents / random noise
    if edge_density > 0.10:
        raise ValueError(ERROR_MESSAGE)

    # 3. Reject non-skin objects with high Z-score deviation from HAM10000 dataset
    z_score = float(np.abs((features - REF_MEAN) / np.maximum(REF_STD, 1e-6)).sum())
    if z_score > 10.0:
        raise ValueError(ERROR_MESSAGE)
