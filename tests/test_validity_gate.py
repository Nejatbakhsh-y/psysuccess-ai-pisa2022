import pandas as pd

from src.validity_gate import apply_validity_gate


def test_validity_gate_flags():
    df = pd.DataFrame({
        "valid_item_count": [10, 2],
        "rasch_theta_se": [0.5, 2.0],
        "predicted_success_probability": [0.8, 0.51],
    })
    out = apply_validity_gate(df, min_valid_item_count=5, max_irt_se=1.0, probability_margin=0.05)
    assert out.loc[0, "validity_gate_pass"] is True or bool(out.loc[0, "validity_gate_pass"]) is True
    assert bool(out.loc[1, "validity_gate_pass"]) is False
