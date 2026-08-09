import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    app_name: str = "skin-cancer-backend"
    host: str = "0.0.0.0"
    port: int = int(os.environ.get("PORT", "5000"))
    model_path: str = "models/best_model.pth"
    device: str = "cpu"
    allowed_origins: list[str] = ["*"]
    max_image_bytes: int = 5_000_000
    rate_limit: str = "120/minute"


settings = Settings()

MODEL_PATH = Path(settings.model_path)
