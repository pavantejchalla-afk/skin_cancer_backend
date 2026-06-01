from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, UploadFile, status

from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.prediction import BatchPredictionResponse, ClassListResponse, HealthResponse, SinglePredictionResponse
from app.services.class_labels import CLASS_LABELS
from app.services.image_validation import ERROR_MESSAGE, validate_image_bytes

router = APIRouter()
logger = get_logger(__name__)

ALLOWED_TYPES = {"image/jpeg", "image/png"}


async def read_upload_file(file: UploadFile) -> bytes:
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only JPG and PNG images are supported")

    data = await file.read()
    if len(data) > settings.max_image_bytes:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="Image exceeds maximum allowed size")

    if len(data) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty image upload")

    try:
        validate_image_bytes(data)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return data


@router.post("/api/predict", response_model=SinglePredictionResponse, status_code=status.HTTP_200_OK)
async def predict_single_image(request: Request, file: UploadFile):
    image_bytes = await read_upload_file(file)
    inference = getattr(request.app.state, "inference", None)
    if inference is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model is not initialized")

    prediction = inference.predict_image_bytes(image_bytes, filename=file.filename or "upload")
    return SinglePredictionResponse(**prediction)


@router.post("/api/predict/batch", response_model=BatchPredictionResponse, status_code=status.HTTP_200_OK)
async def predict_batch(request: Request, files: list[UploadFile]):
    if len(files) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="At least one image is required")

    if len(files) > 5:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Maximum of 5 images per batch request")

    inference = getattr(request.app.state, "inference", None)
    if inference is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model is not initialized")

    predictions = []
    for file in files:
        image_bytes = await read_upload_file(file)
        prediction = inference.predict_image_bytes(image_bytes, filename=file.filename or "upload")
        predictions.append(SinglePredictionResponse(**prediction))

    return BatchPredictionResponse(predictions=predictions)


@router.get("/api/classes", response_model=ClassListResponse)
async def get_classes():
    classes = [
        {
            "code": code,
            "name": metadata["name"],
            "description": metadata["description"],
        }
        for code, metadata in CLASS_LABELS.items()
    ]
    return ClassListResponse(classes=classes)


@router.get("/api/health", response_model=HealthResponse)
async def health_check(request: Request):
    model_loaded = hasattr(request.app.state, "inference")
    return HealthResponse(
        status="ok",
        model_loaded=model_loaded,
        classes=len(CLASS_LABELS),
    )
