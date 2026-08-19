from dataclasses import dataclass

import numpy as np
import pandas as pd

SIGNALS = ("heart_rate", "spo2", "respiration_rate", "temperature")
PHYSIOLOGICAL_BOUNDS = {
    "heart_rate": (20.0, 240.0),
    "spo2": (50.0, 100.0),
    "respiration_rate": (2.0, 70.0),
    "temperature": (30.0, 43.0),
}


@dataclass
class PreprocessedWindow:
    frame: pd.DataFrame
    observed: pd.DataFrame
    quality: pd.DataFrame
    standardized: pd.DataFrame


def _robust_center_scale(values: pd.Series) -> tuple[float, float]:
    valid = values.dropna().to_numpy()
    if len(valid) == 0:
        return 0.0, 1.0
    center = float(np.median(valid))
    mad = float(np.median(np.abs(valid - center)))
    scale = max(1.4826 * mad, float(np.std(valid)) * 0.25, 1e-3)
    return center, scale


def align_and_clean(frame: pd.DataFrame, target_frequency_hz: float = 1.0) -> PreprocessedWindow:
    required = {"timestamp", *SIGNALS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if target_frequency_hz <= 0:
        raise ValueError("target_frequency_hz must be positive")

    source = frame[list(required)].copy().sort_values("timestamp")
    source = source.drop_duplicates("timestamp", keep="last").set_index("timestamp")
    step = 1.0 / target_frequency_hz
    grid = np.arange(float(source.index.min()), float(source.index.max()) + step * 0.5, step)
    aligned = source.reindex(source.index.union(grid)).interpolate(method="index").reindex(grid)
    aligned.index.name = "timestamp"

    observed = pd.DataFrame(False, index=aligned.index, columns=SIGNALS)
    quality = pd.DataFrame(0.0, index=aligned.index, columns=SIGNALS)
    standardized = pd.DataFrame(index=aligned.index, columns=SIGNALS, dtype=float)

    for signal in SIGNALS:
        original = source[signal].reindex(grid)
        observed[signal] = original.notna()
        low, high = PHYSIOLOGICAL_BOUNDS[signal]
        out_of_bounds = ~aligned[signal].between(low, high) & aligned[signal].notna()
        values = aligned[signal].where(~out_of_bounds)
        center, scale = _robust_center_scale(values.iloc[: max(30, len(values) // 3)])
        residual = (values - center) / scale
        local_jump = residual.diff().abs().fillna(0)
        noisy = local_jump > 5
        values = values.mask(noisy).interpolate(limit=15, limit_direction="both")
        values = values.clip(low, high)

        valid = values.notna()
        base_quality = np.where(observed[signal], 1.0, 0.55)
        base_quality = np.where(noisy, 0.2, base_quality)
        base_quality = np.where(out_of_bounds, 0.0, base_quality)
        base_quality = np.where(valid, base_quality, 0.0)
        aligned[signal] = values
        quality[signal] = base_quality

        z = (values - center) / scale
        if signal == "spo2":
            z = -z
        standardized[signal] = z.clip(-8, 8)

    return PreprocessedWindow(
        frame=aligned.reset_index(),
        observed=observed.reset_index(drop=True),
        quality=quality.reset_index(drop=True),
        standardized=standardized.reset_index(drop=True),
    )
