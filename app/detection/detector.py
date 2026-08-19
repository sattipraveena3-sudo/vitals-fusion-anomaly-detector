import numpy as np
import pandas as pd

from app.fusion.state_space import FusionResult
from app.schemas import Anomaly


class VitalAnomalyDetector:
    def __init__(self, fused_threshold: float = 2.5, conflict_threshold: float = 2.2):
        self.fused_threshold = fused_threshold
        self.conflict_threshold = conflict_threshold

    @staticmethod
    def _confidence(severity: float, quality: float) -> float:
        probability = 1.0 / (1.0 + np.exp(-(severity - 1.0)))
        return float(np.clip(probability * (0.55 + 0.45 * quality), 0, 0.999))

    @staticmethod
    def _merge_short_gaps(mask: np.ndarray, max_gap: int = 4) -> np.ndarray:
        merged = mask.copy()
        false_starts = np.flatnonzero(~merged & np.r_[False, merged[:-1]])
        false_ends = np.flatnonzero(~merged & np.r_[merged[1:], False])
        for start, end in zip(false_starts, false_ends, strict=False):
            if start > 0 and end < len(merged) - 1 and end - start + 1 <= max_gap:
                merged[start : end + 1] = True
        return merged

    def detect(
        self,
        frame: pd.DataFrame,
        standardized: pd.DataFrame,
        quality: pd.DataFrame,
        fusion: FusionResult,
    ) -> list[Anomaly]:
        values = frame
        z = standardized.to_numpy(dtype=float)
        dispersion = np.nanstd(z, axis=1)
        quality_mean = quality.mean(axis=1).to_numpy()
        rules = {
            "oxygen_desaturation": (
                values["spo2"].to_numpy() < 92,
                (92 - values["spo2"].to_numpy()) / 4,
            ),
            "heart_rate_irregularity": (
                (values["heart_rate"].to_numpy() < 45)
                | (values["heart_rate"].to_numpy() > 120)
                | (np.abs(z[:, 0]) > 3.5),
                np.maximum(np.abs(z[:, 0]) / 3.5, 0),
            ),
            "abnormal_respiration": (
                (values["respiration_rate"].to_numpy() < 8)
                | (values["respiration_rate"].to_numpy() > 24),
                np.abs(z[:, 2]) / 3,
            ),
            "temperature_elevation": (
                values["temperature"].to_numpy() > 38,
                (values["temperature"].to_numpy() - 37.5) / 0.8,
            ),
            "sensor_conflict": (
                dispersion > self.conflict_threshold,
                dispersion / self.conflict_threshold,
            ),
            "fused_instability": (
                np.abs(fusion.smoothed_state) > self.fused_threshold,
                np.abs(fusion.smoothed_state) / self.fused_threshold,
            ),
        }
        events: list[Anomaly] = []
        timestamps = values["timestamp"].to_numpy()
        for event_type, (mask, severity) in rules.items():
            mask = self._merge_short_gaps(np.asarray(mask) & np.isfinite(severity))
            starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
            ends = np.flatnonzero(mask & ~np.r_[mask[1:], False])
            for start, end in zip(starts, ends, strict=False):
                peak = float(np.nanmax(severity[start : end + 1]))
                mean_quality = float(np.mean(quality_mean[start : end + 1]))
                events.append(
                    Anomaly(
                        start_time=float(timestamps[start]),
                        end_time=float(timestamps[end]),
                        anomaly_type=event_type,
                        confidence=round(self._confidence(peak, mean_quality), 4),
                        severity=round(peak, 4),
                        evidence={
                            "duration_seconds": round(
                                float(timestamps[end] - timestamps[start]), 2
                            ),
                            "mean_data_quality": round(mean_quality, 3),
                        },
                    )
                )
        return sorted(events, key=lambda event: (event.start_time, -event.confidence))
