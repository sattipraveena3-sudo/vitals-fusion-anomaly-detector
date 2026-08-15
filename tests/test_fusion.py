import numpy as np
import pandas as pd

from app.fusion.state_space import RobustStateSpaceFusion


def test_fusion_output_shape_and_finiteness():
    observations = pd.DataFrame(np.zeros((25,4)), columns=list("abcd"))
    observations.loc[10:15,"a"] = 3
    quality = pd.DataFrame(np.ones((25,4)), columns=list("abcd"))
    result = RobustStateSpaceFusion().fuse(observations, quality)
    assert result.smoothed_state.shape == (25,)
    assert np.isfinite(result.smoothed_state).all()
    assert (result.uncertainty > 0).all()


def test_low_quality_outlier_is_downweighted():
    clean = pd.DataFrame(np.zeros((12,4)))
    noisy = clean.copy(); noisy.iloc[6,0] = 8
    high_quality = pd.DataFrame(np.ones((12,4)))
    low_quality = high_quality.copy(); low_quality.iloc[6,0] = 0.01
    model = RobustStateSpaceFusion()
    high = model.fuse(noisy, high_quality).smoothed_state[6]
    low = model.fuse(noisy, low_quality).smoothed_state[6]
    assert abs(low) < abs(high)
