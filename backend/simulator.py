"""Multi-threaded polling of station load. Each station read runs in its
own worker thread via ThreadPoolExecutor, simulating concurrent reads from
independent charging stations (each would normally be a separate network
call to a station's API/telemetry endpoint).
"""
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

STATIONS = [
    {"id": "TN-01", "name": "Shell Recharge", "location": "Guduvanchery", "capacity": 55, "peak_start": 18, "peak_end": 22},
    {"id": "TN-02", "name": "Tata Power EV", "location": "Maraimalai Nagar", "capacity": 50, "peak_start": 19, "peak_end": 21},
    {"id": "TN-03", "name": "Ather Grid", "location": "Tambaram", "capacity": 60, "peak_start": 17, "peak_end": 21},
    {"id": "TN-04", "name": "Zeon Charging", "location": "NH45 Highway", "capacity": 45, "peak_start": 20, "peak_end": 21},
    {"id": "TN-05", "name": "Local EV Station", "location": "Urapakkam", "capacity": 40, "peak_start": 18, "peak_end": 20},
    {"id": "TN-06", "name": "Private Charger", "location": "Vandalur", "capacity": 30, "peak_start": 18, "peak_end": 21},
]


def _load_factor(hour, station, rng):
    is_peak = station["peak_start"] <= hour <= station["peak_end"]
    is_shoulder = (station["peak_start"] - 2 <= hour < station["peak_start"]) or (
        station["peak_end"] < hour <= station["peak_end"] + 1
    )
    if is_peak:
        return 0.85 + rng.uniform(0, 0.12)
    if is_shoulder:
        return 0.50 + rng.uniform(0, 0.20)
    if 6 <= hour < 10:
        return 0.30 + rng.uniform(0, 0.15)
    if 10 <= hour < 17:
        return 0.20 + rng.uniform(0, 0.10)
    return 0.05 + rng.uniform(0, 0.05)


def _poll_station(station, hour, simulate_latency):
    """Stands in for a network call to one station's telemetry endpoint."""
    rng = random.Random(f"{station['id']}-{hour}-{time.time_ns()}")
    if simulate_latency:
        time.sleep(rng.uniform(0.05, 0.2))  # I/O-bound wait, like a real HTTP round trip
    factor = _load_factor(hour, station, rng)
    noise = (rng.random() - 0.5) * station["capacity"] * 0.06
    load = max(0.0, min(station["capacity"], factor * station["capacity"] + noise))
    utilization = load / station["capacity"]
    risk = (
        "critical" if utilization > 0.88
        else "high" if utilization > 0.72
        else "medium" if utilization > 0.50
        else "normal"
    )
    return {**station, "current_load": round(load, 2), "utilization": round(utilization, 4), "risk": risk}


def poll_all_stations(hour, stations=None, simulate_latency=True, max_workers=8):
    """Polls every station concurrently and returns results in station order.

    Returns (results, elapsed_seconds). With simulate_latency on, this is
    measurably faster than polling the stations one at a time.
    """
    stations = stations if stations is not None else STATIONS
    start = time.perf_counter()
    results_by_id = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_poll_station, s, hour, simulate_latency): s["id"] for s in stations}
        for future in as_completed(futures):
            station_id = futures[future]
            results_by_id[station_id] = future.result()
    elapsed = time.perf_counter() - start
    ordered = [results_by_id[s["id"]] for s in stations]
    return ordered, elapsed
