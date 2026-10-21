from fastapi.testclient import TestClient

from backend.main import app


def test_daily_report_rejects_path_traversal():
    client = TestClient(app)
    response = client.get("/api/report/daily", params={"date": "../../README"})
    assert response.status_code == 400


def test_daily_report_rejects_invalid_calendar_date():
    client = TestClient(app)
    response = client.get("/api/report/daily", params={"date": "2026-13-40"})
    assert response.status_code == 400
