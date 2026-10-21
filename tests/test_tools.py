import pandas as pd

import backend.engine as engine
import backend.tools.implementations as implementations


def test_forecast_tool_passes_realtime_data(monkeypatch):
    calls = {}

    monkeypatch.setattr(implementations, "_get_df", lambda: pd.DataFrame())
    def fake_realtime(city):
        calls["city"] = city
        return {"aqi": 42}

    monkeypatch.setattr(implementations.realtime, "fetch_realtime", fake_realtime)
    monkeypatch.setattr(
        engine,
        "get_forecast_alerts",
        lambda df, pollutant, city, live=None: [{"live": live, "city": city}],
    )
    result = implementations.tool_get_forecast("beijing", "PM25")
    assert calls["city"] == "beijing"
    assert result["live"] == {"aqi": 42}
