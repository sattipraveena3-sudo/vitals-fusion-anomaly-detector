from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FusionResult:
    filtered_state: np.ndarray
    smoothed_state: np.ndarray
    uncertainty: np.ndarray
    innovation: np.ndarray


class RobustStateSpaceFusion:
    """One-dimensional latent deviation model with quality-aware observations and RTS smoothing."""

    def __init__(self, process_variance: float = 0.08, measurement_variance: float = 0.35):
        if process_variance <= 0 or measurement_variance <= 0:
            raise ValueError("variances must be positive")
        self.q = process_variance
        self.r = measurement_variance

    def fuse(self, observations: pd.DataFrame, quality: pd.DataFrame) -> FusionResult:
        y = observations.to_numpy(dtype=float)
        ql = quality.to_numpy(dtype=float)
        if y.shape != ql.shape or y.ndim != 2:
            raise ValueError("observations and quality must be equal two-dimensional arrays")
        n = len(y)
        filtered = np.zeros(n)
        variance = np.zeros(n)
        predicted = np.zeros(n)
        predicted_var = np.zeros(n)
        innovation = np.zeros(n)
        state, state_var = 0.0, 1.0

        for t in range(n):
            predicted[t] = state
            predicted_var[t] = state_var + self.q
            valid = np.isfinite(y[t]) & (ql[t] > 0)
            if valid.any():
                weights = ql[t, valid] / self.r
                observation = float(np.average(y[t, valid], weights=weights))
                obs_var = float(1.0 / weights.sum())
                delta = observation - predicted[t]
                huber_weight = min(1.0, 2.5 / max(abs(delta), 1e-9))
                obs_var /= huber_weight
                gain = predicted_var[t] / (predicted_var[t] + obs_var)
                state = predicted[t] + gain * delta
                state_var = (1 - gain) * predicted_var[t]
                innovation[t] = delta
            else:
                state, state_var = predicted[t], predicted_var[t]
            filtered[t], variance[t] = state, state_var

        smoothed = filtered.copy()
        smooth_var = variance.copy()
        for t in range(n - 2, -1, -1):
            gain = variance[t] / max(predicted_var[t + 1], 1e-9)
            smoothed[t] = filtered[t] + gain * (smoothed[t + 1] - predicted[t + 1])
            smooth_var[t] = variance[t] + gain**2 * (smooth_var[t + 1] - predicted_var[t + 1])

        return FusionResult(filtered, smoothed, np.sqrt(np.maximum(smooth_var, 1e-9)), innovation)
