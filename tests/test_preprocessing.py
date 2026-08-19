import numpy as np
import pandas as pd

from app.data.preprocessing import align_and_clean


def test_alignment_resamples_and_tracks_imputation():
    frame = pd.DataFrame(
        {
            "timestamp": [0, 2, 4],
            "heart_rate": [70, np.nan, 74],
            "spo2": [98, 97, 96],
            "respiration_rate": [15, 16, 17],
            "temperature": [36.8, 36.9, 37.0],
        }
    )
    result = align_and_clean(frame, target_frequency_hz=1)
    assert result.frame["timestamp"].tolist() == [0, 1, 2, 3, 4]
    assert np.isfinite(result.frame["heart_rate"]).all()
    assert result.quality.loc[1, "heart_rate"] < 1


def test_out_of_range_reading_gets_low_quality():
    frame = pd.DataFrame(
        {
            "timestamp": [0, 1, 2],
            "heart_rate": [70, 500, 72],
            "spo2": [98, 98, 98],
            "respiration_rate": [15, 15, 15],
            "temperature": [36.8, 36.8, 36.8],
        }
    )
    result = align_and_clean(frame)
    assert result.quality.loc[1, "heart_rate"] == 0
