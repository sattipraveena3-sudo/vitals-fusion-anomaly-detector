from pydantic import BaseModel, Field, model_validator


class SignalWindow(BaseModel):
    timestamps: list[float]
    heart_rate: list[float | None]
    spo2: list[float | None]
    respiration_rate: list[float | None]
    temperature: list[float | None]

    @model_validator(mode="after")
    def validate_lengths(self):
        lengths = {
            len(self.timestamps),
            len(self.heart_rate),
            len(self.spo2),
            len(self.respiration_rate),
            len(self.temperature),
        }
        if len(lengths) != 1:
            raise ValueError("timestamps and all signals must have equal lengths")
        if not self.timestamps:
            raise ValueError("signal window cannot be empty")
        if any(b <= a for a, b in zip(self.timestamps, self.timestamps[1:])):
            raise ValueError("timestamps must be strictly increasing")
        return self


class AnalyzeRequest(BaseModel):
    signals: SignalWindow | None = None
    sample_id: str | None = Field(default=None, pattern="^synthetic-default$")
    target_frequency_hz: float = Field(default=1.0, gt=0, le=10)

    @model_validator(mode="after")
    def choose_source(self):
        if self.signals is None and self.sample_id is None:
            self.sample_id = "synthetic-default"
        if self.signals is not None and self.sample_id is not None:
            raise ValueError("provide signals or sample_id, not both")
        return self


class Anomaly(BaseModel):
    start_time: float
    end_time: float
    anomaly_type: str
    confidence: float
    severity: float
    evidence: dict[str, float | str]


class AnalyzeResponse(BaseModel):
    timestamps: list[float]
    aligned_signals: dict[str, list[float | None]]
    fused_state: list[float]
    uncertainty: list[float]
    quality: dict[str, list[float]]
    anomalies: list[Anomaly]
    summary: dict[str, int | float]
    disclaimer: str = (
        "Research and portfolio demonstration only. This software is not a medical device "
        "and must not be used for diagnosis or clinical decisions."
    )
