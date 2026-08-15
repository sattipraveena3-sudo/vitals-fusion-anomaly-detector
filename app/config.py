from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Multimodal Vital-Signs Anomaly Detector"
    default_sample_seconds: int = 600
    default_frequency_hz: float = 1.0
    max_samples: int = 20_000
    process_variance: float = 0.08
    base_measurement_variance: float = 0.35
    fused_threshold: float = 2.5
    conflict_threshold: float = 2.2

    model_config = SettingsConfigDict(env_file=".env", env_prefix="VITALS_", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
