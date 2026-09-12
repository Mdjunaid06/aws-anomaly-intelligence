"""Backend configuration - load from environment."""

import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Environment
    environment: str = "development"
    debug: bool = False

    # Database
    database_url: str = "postgresql://user:password@localhost/aws_anomaly"
    database_echo: bool = False  # Log SQL queries in debug mode

    # ML Pipeline
    ml_root: Path = Path(__file__).parent.parent.parent.parent / "ml"
    ml_models_dir: Path = ml_root / "artifacts" / "models"
    ml_data_dir: Path = Path(__file__).parent.parent.parent.parent / "data"

    # API
    api_title: str = "AWS Anomaly Intelligence API"
    api_version: str = "1.0.0"
    api_description: str = "ML-powered anomaly detection for AWS weather stations"

    # GenAI (optional)
    genai_enabled: bool = False
    genai_provider: Optional[str] = None  # "openai", "anthropic", etc.
    genai_api_key: Optional[str] = None
    genai_model: Optional[str] = None
    frontend_origin: str = "http://localhost:5173"

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"  # json or text

    class Config:
        """Pydantic settings config."""

        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()
