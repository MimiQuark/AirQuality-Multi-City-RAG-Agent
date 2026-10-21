import numpy as np
import pandas as pd

from backend.engine import add_features


def test_roll6h_does_not_include_current_target():
    rows = 50
    frame = pd.DataFrame(
        {
            "city": ["beijing"] * rows,
            "timestamp": pd.date_range("2026-01-01", periods=rows, freq="h"),
            "month": [1] * rows,
            "AQI": np.arange(rows) + 20,
            "PM25": np.arange(rows) + 10.0,
            "PM10": np.arange(rows) + 20.0,
            "SO2": np.arange(rows) + 1.0,
            "NO2": np.arange(rows) + 5.0,
            "CO": np.arange(rows) + 0.5,
            "O3": np.arange(rows) + 3.0,
            "temperature": np.arange(rows) + 5.0,
            "humidity": np.arange(rows) + 40.0,
            "wind_speed": np.arange(rows) + 1.0,
        }
    )
    featured = add_features(frame)
    row = featured[featured.index == 30].iloc[0]
    expected = frame.loc[24:29, "PM25"].mean()
    assert np.isclose(row["PM25_roll6h"], expected)
    # Current target (index 30) must not be part of the rolling window.
    assert not np.isclose(row["PM25_roll6h"], frame.loc[25:30, "PM25"].mean())
