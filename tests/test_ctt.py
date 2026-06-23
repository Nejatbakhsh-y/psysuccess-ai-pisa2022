import numpy as np
import pandas as pd

from src.ctt import item_difficulty, kr20, student_raw_scores


def test_item_difficulty():
    x = pd.DataFrame({"i1": [1, 0, 1], "i2": [0, 0, 1]})
    diff = item_difficulty(x)
    assert np.isclose(diff["i1"], 2 / 3)
    assert np.isclose(diff["i2"], 1 / 3)


def test_student_raw_scores():
    x = pd.DataFrame({"i1": [1, 0, 1], "i2": [0, np.nan, 1]})
    scores = student_raw_scores(x)
    assert scores.loc[0, "raw_score"] == 1
    assert scores.loc[1, "valid_item_count"] == 1
    assert scores.loc[2, "percent_correct"] == 1.0


def test_kr20_runs():
    x = pd.DataFrame({"i1": [1, 0, 1, 0], "i2": [1, 1, 0, 0], "i3": [1, 0, 1, 1]})
    value = kr20(x)
    assert np.isfinite(value)
