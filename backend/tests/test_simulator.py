from simulator import STATIONS, poll_all_stations


def test_polls_every_station_once():
    results, _ = poll_all_stations(hour=19, simulate_latency=False)
    assert {r["id"] for r in results} == {s["id"] for s in STATIONS}
    assert len(results) == len(STATIONS)


def test_each_reading_is_internally_consistent():
    results, _ = poll_all_stations(hour=19, simulate_latency=False)
    for r in results:
        assert 0 <= r["current_load"] <= r["capacity"]
        assert abs(r["utilization"] - r["current_load"] / r["capacity"]) < 1e-3
        assert r["risk"] in {"normal", "medium", "high", "critical"}


def test_concurrent_polling_is_faster_than_sequential_latency():
    _, elapsed = poll_all_stations(hour=19, simulate_latency=True)
    # Each station sleeps up to 0.2s; sequential would take len(STATIONS) * 0.2s.
    assert elapsed < 0.2 * len(STATIONS)
