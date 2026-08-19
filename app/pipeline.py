import numpy as np
import pandas as pd

from app.config import Settings
from app.data.preprocessing import SIGNALS, align_and_clean
from app.data.synthetic import SyntheticConfig, generate_synthetic_vitals
from app.detection.detector import VitalAnomalyDetector
from app.fusion.state_space import RobustStateSpaceFusion
from app.schemas import AnalyzeRequest, AnalyzeResponse


class AnalysisPipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.fusion = RobustStateSpaceFusion(
            settings.process_variance, settings.base_measurement_variance
        )
        self.detector = VitalAnomalyDetector(settings.fused_threshold, settings.conflict_threshold)

    def analyze(self, request: AnalyzeRequest) -> AnalyzeResponse:
        if request.signals is not None:
            frame = pd.DataFrame(request.signals.model_dump())
        else:
            frame = generate_synthetic_vitals(
                SyntheticConfig(duration_seconds=self.settings.default_sample_seconds)
            )
        if len(frame) > self.settings.max_samples:
            raise ValueError(f"window exceeds maximum of {self.settings.max_samples} samples")
        processed = align_and_clean(frame, request.target_frequency_hz)
        result = self.fusion.fuse(processed.standardized, processed.quality)
        anomalies = self.detector.detect(
            processed.frame, processed.standardized, processed.quality, result
        )
        aligned = {
            signal: [
                None if not np.isfinite(value) else round(float(value), 4)
                for value in processed.frame[signal]
            ]
            for signal in SIGNALS
        }
        return AnalyzeResponse(
            timestamps=[round(float(value), 4) for value in processed.frame["timestamp"]],
            aligned_signals=aligned,
            fused_state=np.round(result.smoothed_state, 5).tolist(),
            uncertainty=np.round(result.uncertainty, 5).tolist(),
            quality={signal: np.round(processed.quality[signal], 3).tolist() for signal in SIGNALS},
            anomalies=anomalies,
            summary={
                "samples": len(processed.frame),
                "anomaly_events": len(anomalies),
                "mean_quality": round(float(processed.quality.to_numpy().mean()), 4),
            },
        )
