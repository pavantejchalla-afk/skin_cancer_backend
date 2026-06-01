from io import BytesIO

import numpy as np
from PIL import Image

from app.services.image_validation import validate_image_bytes


def _png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_accepts_real_skin_reference_image():
    image = Image.open("../archive/HAM10000_images_part_1/ISIC_0024306.jpg").convert("RGB")
    validate_image_bytes(_png_bytes(image))


def test_rejects_solid_color_non_skin_image():
    image = Image.fromarray(np.full((224, 224, 3), 255, dtype=np.uint8), "RGB")

    try:
        validate_image_bytes(_png_bytes(image))
    except ValueError as exc:
        assert str(exc) == "upload the skin disease image correctky"
    else:
        raise AssertionError("Expected solid-color image to be rejected")


def test_rejects_invalid_image_bytes():
    try:
        validate_image_bytes(b"not-a-real-image")
    except ValueError as exc:
        assert str(exc) == "upload the skin disease image correctky"
    else:
        raise AssertionError("Expected invalid bytes to be rejected")
