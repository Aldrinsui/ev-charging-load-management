import pytest

import app as app_module


@pytest.fixture
def client():
    flask_app = app_module.create_app(database_url="sqlite:///:memory:")
    return flask_app.test_client()


def test_stations_endpoint_polls_concurrently_and_persists(client):
    resp = client.get("/api/stations?hour=19")
    assert resp.status_code == 200
    body = resp.get_json()
    assert len(body["stations"]) == 6
    assert "pollElapsedMs" in body


def test_forecast_endpoint_returns_requested_steps(client):
    resp = client.post("/api/forecast", json={
        "stationId": "TN-01",
        "history": [10, 12, 14, 13, 15, 16, 18],
        "steps": 4,
    })
    assert resp.status_code == 200
    assert len(resp.get_json()["forecast"]) == 4


def test_forecast_rejects_short_history(client):
    resp = client.post("/api/forecast", json={"history": [1, 2]})
    assert resp.status_code == 400


def test_model_metrics_endpoint_reports_real_numbers(client):
    resp = client.get("/api/model-metrics")
    assert resp.status_code == 200
    body = resp.get_json()
    assert "mape" in body and body["mape"] > 0
