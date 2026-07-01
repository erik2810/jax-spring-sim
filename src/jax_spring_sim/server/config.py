"""Server configuration via pydantic-settings (env-overridable, ``JSS_`` prefix)."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="JSS_", env_file=".env", extra="ignore")

    host: str = "127.0.0.1"
    port: int = 8000
    # Vite dev server origins allowed for the health/info endpoints (CORS).
    allowed_origins: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]
    default_fps: float = 60.0


settings = Settings()
