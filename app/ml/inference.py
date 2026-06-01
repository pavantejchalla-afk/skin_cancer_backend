from __future__ import annotations

from io import BytesIO
from pathlib import Path

import torch
from PIL import Image
from torchvision import models
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from app.core.config import settings
from app.core.logging import get_logger
from app.services.class_labels import CLASS_LABELS, DISCLAIMER

logger = get_logger(__name__)


class SkinCancerInference:
    def __init__(self, model_path: str | None = None, device: str | None = None):
        self.model_path = Path(model_path or settings.model_path)
        self.device = torch.device(device or settings.device)
        self.model = None
        self.classes = list(CLASS_LABELS.keys())
        self.transform = Compose(
            [
                Resize((224, 224)),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def build_model(self) -> torch.nn.Module:
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        model.classifier[1] = torch.nn.Linear(in_features=model.classifier[1].in_features, out_features=len(self.classes))
        return model

    def load_model(self):
        model = self.build_model()

        if self.model_path and self.model_path.exists():
            state = torch.load(self.model_path, map_location=self.device)
            if isinstance(state, dict) and "model_state_dict" in state:
                model.load_state_dict(state["model_state_dict"])
                logger.info("Loaded model weights from %s", self.model_path)
            else:
                model.load_state_dict(state)
                logger.info("Loaded model state dict from %s", self.model_path)
        else:
            logger.warning("Model checkpoint not found at %s. Using ImageNet-pretrained MobileNetV2 with randomly initialized classification head.", self.model_path)

        model.to(self.device)
        model.eval()
        self.model = model

    def preprocess_image(self, image_bytes: bytes) -> torch.Tensor:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        return tensor

    def predict_tensor(self, tensor: torch.Tensor):
        if self.model is None:
            raise RuntimeError("Model is not loaded")

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        ranked = sorted(enumerate(probs), key=lambda item: item[1], reverse=True)
        top3 = []
        for idx, prob in ranked[:3]:
            label = self.classes[idx]
            top3.append(
                {
                    "label": label,
                    "confidence": round(float(prob * 100), 2),
                }
            )

        top_label = top3[0]["label"]
        return {
            "predicted_label": top_label,
            "predicted_class": CLASS_LABELS[top_label]["name"],
            "confidence": top3[0]["confidence"],
            "top_3": top3,
            "disclaimer": DISCLAIMER,
        }

    def predict_image_bytes(self, image_bytes: bytes, filename: str = "image"):
        tensor = self.preprocess_image(image_bytes)
        prediction = self.predict_tensor(tensor)
        return {
            "filename": filename,
            "predicted_label": prediction["predicted_label"],
            "predicted_class": prediction["predicted_class"],
            "confidence": prediction["confidence"],
            "top_3": prediction["top_3"],
            "disclaimer": prediction["disclaimer"],
        }
