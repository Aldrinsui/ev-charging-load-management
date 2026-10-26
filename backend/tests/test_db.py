import pytest

import db
from simulator import STATIONS, poll_all_stations


@pytest.fixture
def session():
    engine = db.get_engine("sqlite:///:memory:")
    Session = db.init_db(engine)
    s = Session()
    yield s
    s.close()


def test_upsert_stations_is_idempotent(session):
    db.upsert_stations(session, STATIONS)
    db.upsert_stations(session, STATIONS)  # should not create duplicates
    assert session.query(db.Station).count() == len(STATIONS)


def test_record_readings_persists_all_stations(session):
    db.upsert_stations(session, STATIONS)
    results, _ = poll_all_stations(hour=19, simulate_latency=False)
    db.record_readings(session, hour=19, poll_results=results)
    assert session.query(db.Reading).count() == len(STATIONS)
    row = session.query(db.Reading).first()
    assert row.hour == 19
    assert row.risk in {"normal", "medium", "high", "critical"}


def test_record_forecast_stores_metrics(session):
    db.upsert_stations(session, STATIONS)
    db.record_forecast(session, "TN-01", predicted_kw=42.5, metrics={"mae": 1.0, "rmse": 1.5, "mape": 10.0})
    row = session.query(db.ForecastRun).first()
    assert row.station_id == "TN-01"
    assert row.predicted_kw == 42.5
    assert row.mape == 10.0
