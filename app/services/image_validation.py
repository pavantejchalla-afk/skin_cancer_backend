from __future__ import annotations

from functools import lru_cache
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

ERROR_MESSAGE = "Please upload a clear skin lesion image"
ARCHIVE_ROOT = Path(__file__).resolve().parents[2].parent / "archive"
REFERENCE_LIMIT = 40
IMAGE_SIZE = (224, 224)


@lru_cache(maxsize=1)
def load_reference_stats() -> tuple[np.ndarray, np.ndarray] | None:
    image_paths = []
    for folder in (ARCHIVE_ROOT / "HAM10000_images_part_1", ARCHIVE_ROOT / "HAM10000_images_part_2"):
        if folder.exists():
            image_paths.extend(sorted(folder.glob("*.jpg")))

    if not image_paths:
        return None

    samples = image_paths[:REFERENCE_LIMIT]
    features = []

    for path in samples:
        try:
            with Image.open(path) as image:
                image = image.convert("RGB").resize(IMAGE_SIZE)
                features.append(compute_feature_vector(image))
        except Exception:
            continue

    if not features:
        return None

    stacked = np.stack(features)
    return stacked.mean(axis=0), stacked.std(axis=0)


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
            if image.size[0] < 128 or image.size[1] < 128:
                raise ValueError(ERROR_MESSAGE)

            image = image.resize(IMAGE_SIZE)
            features = compute_feature_vector(image)
    except Exception as exc:
        raise ValueError(ERROR_MESSAGE) from exc

    stats = load_reference_stats()
    if stats is None:
        if float(features[0]) < 0.0015 and float(features[1]) < 0.012 and float(features[3]) < 0.22:
            raise ValueError(ERROR_MESSAGE)
        return

    mean, std = stats
    z_score = np.abs((features - mean) / np.maximum(std, 1e-6))

    if float(z_score.sum()) > 15.0:
        raise ValueError(ERROR_MESSAGE)
