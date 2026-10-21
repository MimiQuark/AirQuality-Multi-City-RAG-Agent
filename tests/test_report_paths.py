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


def test_report_generation_fails_closed_without_admin_token():
    from backend import main

    if main.app_settings.admin_api_token:
        return
    client = TestClient(app)
    response = client.post("/api/report/daily/generate")
    assert response.status_code == 404
