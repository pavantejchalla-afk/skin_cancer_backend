from pydantic import BaseModel, ConfigDict, Field


class PredictionResult(BaseModel):
    label: str
    confidence: float = Field(..., ge=0.0, le=100.0)


class SinglePredictionResponse(BaseModel):
    filename: str
    predicted_label: str
    predicted_class: str
    confidence: float
    top_3: list[PredictionResult]
    disclaimer: str


class BatchPredictionResponse(BaseModel):
    predictions: list[SinglePredictionResponse]


class ClassInfo(BaseModel):
    code: str
    name: str
    description: str


class ClassListResponse(BaseModel):
    classes: list[ClassInfo]


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    classes: int
