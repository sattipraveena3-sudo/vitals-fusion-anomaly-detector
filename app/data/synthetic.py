from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyntheticConfig:
    duration_seconds: int = 600
    seed: int = 17


def generate_synthetic_vitals(config: SyntheticConfig | None = None) -> pd.DataFrame:
    """Create synchronized vitals with known events, missingness, drift, and sensor noise."""
    config = config or SyntheticConfig()
    rng = np.random.default_rng(config.seed)
    time = np.arange(config.duration_seconds, dtype=float)
    slow = np.sin(2 * np.pi * time / 180)

    heart_rate = 72 + 4 * slow + rng.normal(0, 1.4, len(time))
    spo2 = 97.5 - 0.4 * slow + rng.normal(0, 0.25, len(time))
    respiration = 16 + 1.2 * slow + rng.normal(0, 0.45, len(time))
    temperature = 36.8 + 0.08 * np.sin(2 * np.pi * time / 400) + rng.normal(0, 0.025, len(time))

    desaturation = (time >= 155) & (time <= 195)
    spo2[desaturation] -= 8 * np.sin(np.pi * (time[desaturation] - 155) / 40)
    respiration[desaturation] += 5

    irregular = (time >= 315) & (time <= 345)
    heart_rate[irregular] += 28 * np.sin(2 * np.pi * time[irregular] / 5)

    fever = time >= 465
    temperature[fever] += np.linspace(0, 1.8, fever.sum())
    heart_rate[fever] += np.linspace(0, 16, fever.sum())

    heart_rate[235 : min(248, len(time))] = np.nan
    spo2[390 : min(410, len(time))] = np.nan
    noise_start, noise_end = 270, min(280, len(time))
    if noise_end > noise_start:
        respiration[noise_start:noise_end] += rng.normal(0, 8, noise_end - noise_start)
    corrupt_start, corrupt_end = 420, min(428, len(time))
    if corrupt_end > corrupt_start:
        spo2[corrupt_start:corrupt_end] = 82 + rng.normal(0, 5, corrupt_end - corrupt_start)

    return pd.DataFrame(
        {
            "timestamp": time,
            "heart_rate": heart_rate,
            "spo2": spo2,
            "respiration_rate": respiration,
            "temperature": temperature,
        }
    )
