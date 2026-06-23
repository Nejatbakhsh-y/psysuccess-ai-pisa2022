import numpy as np
import pandas as pd

from src.irt import fit_rasch_jml


def test_rasch_shapes():
    x = pd.DataFrame({
        "i1": [1, 1, 0, 0],
        "i2": [1, 0, 0, 0],
        "i3": [1, 1, 1, 0],
    })
    result = fit_rasch_jml(x, max_iter=5)
    assert result.theta.shape[0] == 4
    assert result.item_difficulty.shape[0] == 3
    assert np.isfinite(result.theta).all()
