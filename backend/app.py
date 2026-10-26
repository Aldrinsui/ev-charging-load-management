"""Flask API for the EV charging load management dashboard."""
import os
from datetime import datetime

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request

import db
import llm
from prompts import build_recommendation_prompt
from simulator import STATIONS, poll_all_stations

load_dotenv()

VALID_RISK = {"normal", "medium", "high", "critical"}
_lstm_model = None  # lazy-loaded so importing app.py doesn't require trained weights


def _get_lstm():
    global _lstm_model
    if _lstm_model is None:
        from ml.lstm_numpy import LSTM
        from ml.train import WEIGHTS_PATH
        _lstm_model = LSTM.load(WEIGHTS_PATH)
    return _lstm_model


def create_app(database_url=None):
    app = Flask(__name__)
    engine = db.get_engine(database_url)
    app.Session = db.init_db(engine)
    session = app.Session()
    db.upsert_stations(session, STATIONS)
    session.close()

    @app.after_request
    def add_cors_headers(response):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
        response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
        return response

    @app.before_request
    def handle_preflight():
        if request.method == "OPTIONS":
            res = Response()
            res.headers["Access-Control-Allow-Origin"] = "*"
            res.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
            res.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
            return res

    @app.get("/api/health")
    def health():
        return jsonify(status="ok")

    @app.get("/api/stations")
    def stations():
        """Polls all stations concurrently (multi-threaded) and persists the readings."""
        hour = request.args.get("hour", default=datetime.now().hour, type=int)
        results, elapsed_s = poll_all_stations(hour)
        session = app.Session()
        try:
            db.record_readings(session, hour, results)
        finally:
            session.close()
        return jsonify(stations=results, pollElapsedMs=round(elapsed_s * 1000, 1))

    @app.post("/api/recommendations")
    def recommendations():
        body = request.get_json(silent=True)
        error = _validate_snapshot(body)
        if error:
            return jsonify(error=error), 400

        stations_in = body["stations"]
        prompt = build_recommendation_prompt(
            stations=stations_in,
            peak_hour=body.get("peakHour", 19),
            avg_util=body["avgUtil"],
            timestamp=datetime.now().strftime("%H:%M:%S"),
        )
        try:
            result = llm.get_recommendations(prompt)
        except llm.LLMError as exc:
            app.logger.warning("LLM failure: %s", exc)
            return jsonify(error="recommendations unavailable"), 502
        return jsonify(result)

    @app.post("/api/forecast")
    def forecast():
        """JSON contract between the ML layer and the rest of the app:
        { "stationId": str, "history": [float, ...] } -> { "forecast": [float, ...], "metrics": {...} }
        """
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return jsonify(error="request body must be JSON"), 400
        history = body.get("history")
        if not isinstance(history, list) or len(history) < 5 or not all(isinstance(v, (int, float)) for v in history):
            return jsonify(error="'history' must be a list of at least 5 numbers"), 400
        steps = body.get("steps", 6)
        if not isinstance(steps, int) or not 1 <= steps <= 24:
            return jsonify(error="'steps' must be an integer between 1 and 24"), 400

        import numpy as np
        from ml.train import normalize

        model = _get_lstm()
        series = np.array(history[-5:], dtype=float)
        norm, mean, std = normalize(series) if series.std() > 0 else (series - series.mean(), series.mean(), 1.0)
        window = norm.reshape(-1, 1)
        forecast_norm = model.forecast(window, steps)
        forecast_vals = [round(float(v * std + mean), 2) for v in forecast_norm]

        if body.get("stationId"):
            session = app.Session()
            try:
                db.record_forecast(session, body["stationId"], forecast_vals[0])
            finally:
                session.close()

        return jsonify(forecast=forecast_vals)

    @app.get("/api/model-metrics")
    def model_metrics():
        """Honest, measured metrics from the last training run (not hardcoded claims)."""
        import json
        metrics_path = os.path.join(os.path.dirname(__file__), "ml", "metrics.json")
        if not os.path.exists(metrics_path):
            return jsonify(error="model has not been trained yet; run `python -m ml.train`"), 404
        with open(metrics_path) as f:
            return jsonify(json.load(f))

    return app


def _validate_snapshot(body):
    if not isinstance(body, dict):
        return "request body must be JSON"
    stations = body.get("stations")
    if not isinstance(stations, list) or not stations:
        return "'stations' must be a non-empty list"
    for s in stations:
        if not isinstance(s, dict):
            return "each station must be an object"
        for key in ("id", "name", "location"):
            if not isinstance(s.get(key), str):
                return f"station field '{key}' must be a string"
        util = s.get("utilization")
        if not isinstance(util, (int, float)) or not 0 <= util <= 1.5:
            return "station 'utilization' must be a number between 0 and 1"
        if s.get("risk") not in VALID_RISK:
            return "station 'risk' must be one of normal, medium, high, critical"
    if not isinstance(body.get("avgUtil"), (int, float)):
        return "'avgUtil' must be a number"
    return None


app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(port=port, debug=True)
